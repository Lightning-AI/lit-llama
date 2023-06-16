"""
Multi-modal instruction-tuning with LLaMA-Adapter v2 on the LLaVA-instruct dataset with images from MS COCO.

LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model
https://arxiv.org/abs/2304.15010

This script runs on a single GPU by default. You can adjust the `micro_batch_size` to fit your GPU memory.
You can finetune within 1 hour as done in the original paper using DeepSpeed Zero-2 on 8 A100 GPUs by setting the
devices variable to `devices = 8` and `micro_batch_size = 8` (or higher).

Additional Requirements to install:

pip install git+https://github.com/openai/CLIP.git torchvision pillow

"""
import os
import sys
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import time
from pathlib import Path
import shutil
import random

import lightning as L
import torch

import clip
from scripts.prepare_alpaca import prepare_sample
from torch.utils.data import Dataset
import json
from lit_llama.tokenizer import Tokenizer
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

from lit_llama.utils import cycle_dataloader
from lit_llama.adapter_v2 import LLaMA, LLaMAConfig, adapter_state_from_state_dict, mark_only_adapter_trainable
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
from lightning.fabric.strategies import DeepSpeedStrategy
from generate import generate

import wandb

eval_interval = 200
save_interval = 400
eval_iters = 100
devices = 1

# Hyperparameters
learning_rate = 9e-3
batch_size = 64 / devices
micro_batch_size = 8
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
epoch_size = 50000  # train dataset size
num_epochs = 1000
max_iters = num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 0.02
max_seq_length = 256  # see scripts/prepare_alpaca.py
warmup_iters = 2 * (epoch_size // micro_batch_size) // devices  # 2 epoch

ds_config = {
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_iters,
    "zero_optimization": {"stage": 2},
}


def main(
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    out_dir: str = "out/adapter_v2/llava-debug",
):

    fabric = L.Fabric(
        accelerator="cuda",
        devices=devices,
        # strategy=(DeepSpeedStrategy(config=ds_config) if devices > 1 else "auto"),
        # strategy="ddp",
        precision="bf16-true",
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        wandb.init(project="adapter-v2-multi-modal", name="llava-clip-transform")

    config = LLaMAConfig(block_size=max_seq_length)

    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(
            f"Can't find the pretrained weights at {pretrained_path}."
            " Please follow the instructions in the README to download them."
        )
    checkpoint = torch.load(pretrained_path)

    with fabric.init_module():
        model = LLaMA(config)
    
    # strict=False because missing keys due to adapter weights not contained in state dict
    model.load_state_dict(checkpoint, strict=False)

    mark_only_adapter_trainable(model)

    dataloader = get_dataloader(batch_size=micro_batch_size, num_workers=2)

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_params}")
    for n, p in model.named_parameters():
        print(n, p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, dataloader, out_dir)

    # Save the final checkpoint at the end of training
    save_model_checkpoint(fabric, model, os.path.join(out_dir, f"lit-llama-adapter-v2.pth"))


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    clip_model,
    optimizer: torch.optim.Optimizer,
    dataloader,
    out_dir: str,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    clip_model, clip_transform = clip.load("ViT-L/14")
    clip_model = fabric.to_device(clip_model)  # keep in default precision

    data_iterator = cycle_dataloader(dataloader)
    step_count = 0

    for iter_num in range(max_iters):

        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        img, input_ids, targets = next(data_iterator)
        img, input_ids, targets = fabric.to_device((img, input_ids, targets))
        with torch.no_grad(), torch.cuda.amp.autocast():
            clip_feats = encode_image(clip_model, img)

        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
            logits = model(input_ids, clip_feats)
            loss = loss_fn(logits, targets)
            fabric.backward(loss / gradient_accumulation_iters)

        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, dataloader)
                # fabric.print(f"iter {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step_count % save_interval == 0:
                print(f"Saving adapter weights to {out_dir}")
                # TODO: Provide a function/script to merge the adapter weights with pretrained weights
                save_model_checkpoint(fabric, model, os.path.join(out_dir, f"iter-{iter_num:06d}.pth"))

        dt = time.time() - t0

        loss = loss.item()
   
        if fabric.global_rank == 0:
            wandb.log({"iter": iter_num, "train_loss": loss, "step": step_count, "lr": lr})
        fabric.print(f"iter {iter_num}, step {step_count}: train_loss {loss:.4f}, time: {dt*1000:.2f}ms")


def generate_response(model, instruction, input="", image_features=None):
    tokenizer = Tokenizer("checkpoints/lit-llama/tokenizer.model")
    sample = {"instruction": instruction, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        image_features=image_features,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
        temperature=0.8,
    )
    output = tokenizer.decode(output)
    return output


predictions_buffer = []

@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, dataloader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    # produce an example:
    clip_model, clip_transform = clip.load("ViT-L/14")
    clip_model = fabric.to_device(clip_model)  # keep in default precision
    
    # pick a random image
    test_folder = "/data/shared/datasets/coco/test2014/"
    file = os.path.join(test_folder, random.choice(os.listdir(test_folder)))
    img = Image.open(file)
    img = clip_transform(img)
    img = img.unsqueeze(0)
    img = img.to(fabric.device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        clip_feats = encode_image(clip_model, img)
    instruction = "Describe the contents of the image."
    output = generate_response(model, instruction, image_features=clip_feats)
    fabric.print(instruction)
    fabric.print(output)

    if fabric.global_rank == 0:
        predictions_buffer.append([wandb.Image(img.cpu()), instruction, output])
        predictions = wandb.Table(columns=["image", "input", "output"], rows=predictions_buffer)
        wandb.log({"predictions": predictions})

    model.train()


def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss


def save_model_checkpoint(fabric, model, file_path):
    file_path = Path(file_path)

    if isinstance(fabric.strategy, DeepSpeedStrategy):
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        tmp_path = file_path.with_suffix(".tmp")
        fabric.save(tmp_path, {"model": model})
        fabric.barrier()
        if fabric.global_rank == 0:
            # Create a consolidated checkpoint with the same name next to the deepspeed checkpoint
            # and only keep the adapter weights
            state_dict = get_fp32_state_dict_from_zero_checkpoint(tmp_path)
            state_dict = adapter_state_from_state_dict(state_dict)
            torch.save(state_dict, file_path)
            shutil.rmtree(tmp_path)
    else:
        state_dict = adapter_state_from_state_dict(model.state_dict())
        if fabric.global_rank == 0:
            torch.save(state_dict, file_path)
        fabric.barrier()


def encode_image(clip_model, x):
    x = clip_model.visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    x = torch.cat([clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
    x = x + clip_model.visual.positional_embedding.to(x.dtype)
    x = clip_model.visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = clip_model.visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.visual.ln_post(x[:, :, :])
    if clip_model.visual.proj is not None:
        x = x @ clip_model.visual.proj
    return x


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=InterpolationMode.BICUBIC, antialias=None),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])


class LLaVAInstruct(Dataset):
    def __init__(
        self, 
        config_path="data/datasets/llava-instruct/llava_instruct_150k.json", 
        coco_root="data/datasets/coco/", 
        tokenizer_path="checkpoints/lit-llama/tokenizer.model",
        max_length=256, 
        img_transform=train_transform,
    ):
        self.coco_root = coco_root
        self.annotations = json.load(open(config_path))
        self.transform = img_transform
        self.max_length = max_length
        self.tokenizer = Tokenizer(tokenizer_path)
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        image_name = annotation['image']
        question = annotation['conversations'][0]['value']
        answer = annotation['conversations'][1]['value']
        filename = os.path.join(self.coco_root, "train2014", f"COCO_train2014_" + image_name)
        image = Image.open(filename).convert('RGB')
        image = self.transform(image)
        example = {"instruction": question, "input": "", "output": answer}
        prepared = prepare_sample(example, self.tokenizer, max_length=self.max_length, mask_inputs=False)
        return {
            "image": image, 
            "input_ids": prepared["input_ids"].type(torch.int64), 
            "labels": prepared["labels"].type(torch.int64),
        }


def collate_fn(samples):
    images = [item["image"] for item in samples]
    input_ids = [item["input_ids"] for item in samples]
    labels = [item["labels"] for item in samples]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    img = torch.stack(images)
    return img, x, y


def get_dataloader(batch_size=1, num_workers=0, img_transform=train_transform):
    dataset = LLaVAInstruct(img_transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI(main)

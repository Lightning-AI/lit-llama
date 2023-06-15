"""
Instruction-tuning with LLaMA-Adapter v2 on the Alpaca dataset following the paper

LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model
https://arxiv.org/abs/2304.15010

This script runs on a single GPU by default. You can adjust the `micro_batch_size` to fit your GPU memory.
You can finetune within 1 hour as done in the original paper using DeepSpeed Zero-2 on 8 A100 GPUs by setting the
devices variable to `devices = 8` and `micro_batch_size = 8` (or higher).

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import os
import sys
import time
from pathlib import Path
import shutil
import random

import lightning as L
import numpy as np
import torch
import torch.nn as nn

import clip
from torch.utils.data import DataLoader
from  torchvision.datasets import CocoCaptions
import torchvision.transforms as transforms
from PIL import Image
import clip
import torch
from scripts.prepare_alpaca import prepare_sample
import wandb

# Requirements
# git+https://github.com/openai/CLIP.git


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# from generate.adapter_v2 import generate
from lit_llama.adapter_v2 import (
    LLaMA, LLaMAConfig,
    adapter_state_from_state_dict,
    mark_instruction_adapter_trainable,
    mark_visual_adapter_trainable,
)
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
from lightning.fabric.strategies import DeepSpeedStrategy
from dataset import get_dataloader



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
    data_dir: str = "data/alpaca", 
    pretrained_path: str = "checkpoints/lit-llama/7B/lit-llama.pth",
    out_dir: str = "out/adapter_v2/llava",
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

    clip_model, clip_transform = clip.load("ViT-L/14")
    clip_model = fabric.to_device(clip_model)  # keep in default precision
    # coco_dataloader = get_coco_dataloader(clip_transform)

    dataloader = get_dataloader(batch_size=micro_batch_size, num_workers=2, img_transform=clip_transform)


    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_params}")
    for n, p in model.named_parameters():
        print(n, p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, clip_model, optimizer, dataloader, out_dir)

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
    tokenizer = Tokenizer("checkpoints/lit-llama/tokenizer.model")
    data_iterator = iter(dataloader)
    step_count = 0

    visual_loss = 10.
    text_loss = 10.

    # mark_visual_adapter_trainable(model)
    mark_adapter_trainable(model)

    for iter_num in range(max_iters):

        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        try:
            img, input_ids, targets = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            img, input_ids, targets = next(data_iterator)

        img, input_ids, targets = fabric.to_device((img, input_ids, targets))
        with torch.no_grad(), torch.cuda.amp.autocast():
            clip_feats = clip_encode_image(clip_model, img)
       

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
    # losses = torch.zeros(eval_iters)
    # data_iterator = iter(dataloader)
    # for k in range(eval_iters):
    #     input_ids, targets = get_batch(fabric, data_iterator)
    #     logits = model(input_ids)
    #     loss = loss_fn(logits, targets)
    #     losses[k] = loss.item()
    # val_loss = losses.mean()

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
        clip_feats = clip_encode_image(clip_model, img)
    instruction = "Describe the contents of the image."
    output = generate_response(model, instruction, image_features=clip_feats)
    fabric.print(instruction)
    fabric.print(output)

    if fabric.global_rank == 0:
        predictions_buffer.append([wandb.Image(img.cpu()), instruction, output])
        predictions = wandb.Table(columns=["image", "input", "output"], rows=predictions_buffer)
        wandb.log({"predictions": predictions})

    model.train()
    # return val_loss.item()

def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss

def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def get_batch_2(fabric: L.Fabric, data: list):
    input_ids = [item["input_ids"].type(torch.int64) for item in data]
    labels = [item["labels"].type(torch.int64) for item in data]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


def get_coco_dataloader(clip_transform):
    pick_caption = transforms.Lambda(lambda c: random.choice(c))
    train_captions = CocoCaptions(
        root="/data/shared/datasets/coco/train2014/",
        annFile="/data/shared/datasets/coco/annotations/captions_train2014.json",
        transform=clip_transform,
        target_transform=pick_caption,
    )
    val_captions = CocoCaptions(
        root="/data/shared/datasets/coco/val2014/",
        annFile="/data/shared/datasets/coco/annotations/captions_val2014.json",
        transform=clip_transform,
        target_transform=pick_caption,
    )
    dataset = train_captions + val_captions
    print('Number of samples: ', len(dataset))
    return DataLoader(dataset, batch_size=micro_batch_size, shuffle=True, num_workers=4)


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



def clip_encode_image(clip_model, x):
    # modified from CLIP
    x = clip_model.visual.conv1(x)  # shape = [*, width, grid, grid]
    # shape = [*, width, grid ** 2]
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + clip_model.visual.positional_embedding.to(x.dtype)
    x = clip_model.visual.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.visual.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    # preserve all spatial tokens
    x = clip_model.visual.ln_post(x[:, :, :])

    if clip_model.visual.proj is not None:
        x = x @ clip_model.visual.proj

    return x



@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    image_features,
    max_new_tokens: int,
    max_seq_length: int,
    temperature: float = 1.0,
    top_k=None,
    eos_id=None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
        eos_id: If specified, stop generating any more token once the <eos> token is triggered
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = idx.size(0)
    T_new = T + max_new_tokens
    empty = torch.empty(T_new, dtype=idx.dtype, device=idx.device)
    empty[:T] = idx
    idx = empty

    # generate max_new_tokens tokens
    for t in range(T, T_new):
        # ignore the not-filled-yet tokens
        idx_cond = idx[:t]
        # if the sequence context is growing too long we must crop it at max_seq_length
        idx_cond = idx_cond if t <= max_seq_length else idx_cond[-max_seq_length:]

        # forward
        logits = model(idx_cond.view(1, -1), image_features)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[[-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # concatenate the new generation
        # https://github.com/pytorch/pytorch/issues/101936
        idx[t] = idx_next.item() if idx.device.type == "mps" else idx_next

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:t + 1]  # include the EOS token

    return idx


import re
INSTRUCTION_ADAPTER_REGEX = ".*transformer.*adapter_wte|.*transformer.*gating_factor|.*transformer.*bias|.*rms_1.*|.*rms_2.*|.*ln_f.*"
VISUAL_ADAPTER_REGEX = ".*clip_proj.*|.*visual_proj.*|.*visual_blocks.*|.*visual_query.*"



def mark_adapter_trainable(model: LLaMA) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = bool(re.match(VISUAL_ADAPTER_REGEX, name) or re.match(INSTRUCTION_ADAPTER_REGEX, name))

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI(main)

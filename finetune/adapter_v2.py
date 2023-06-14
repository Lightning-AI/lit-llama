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

import lightning as L
import numpy as np
import torch
import torch.nn as nn

import clip
from torch.utils.data import DataLoader
# from timm.models.vision_transformer import Block
from  torchvision.datasets import CocoCaptions
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import clip
from timm.models.vision_transformer import Block as ViTBlock
import torch


# Requirements
# git+https://github.com/openai/CLIP.git


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama.adapter_v2 import (
    LLaMA, LLaMAConfig,
    # mark_only_adapter_v2_as_trainable,
    # add_adapter_v2_parameters_to_linear_layers,
    # adapter_v2_state_from_state_dict
    mark_instruction_adapter_trainable
    )
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
from lightning.fabric.strategies import DeepSpeedStrategy


eval_interval = 600
save_interval = 1000
eval_iters = 100
log_interval = 1
devices = 1

# Hyperparameters
learning_rate = 9e-3
batch_size = 64 / devices
micro_batch_size = 1
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
epoch_size = 50000  # train dataset size
num_epochs = 5
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
    out_dir: str = "out/adapter_v2/alpaca",
):

    fabric = L.Fabric(
        accelerator="cuda",
        devices=1,
        strategy=(DeepSpeedStrategy(config=ds_config) if devices > 1 else "auto"),
        precision="bf16-true",
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets(data_dir=data_dir)

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

    # add_adapter_v2_parameters_to_linear_layers(model)
    # mark_only_adapter_v2_as_trainable(model)
    mark_instruction_adapter_trainable(model)

    clip_model, clip_transform = clip.load("ViT-L/14")
    clip_model = fabric.to_device(clip_model)  # keep in default precision
    coco_dataloader = get_coco_dataloader(clip_transform)

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_params}")
    for n, p in model.named_parameters():
        print(n, p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, clip_model, optimizer, train_data, val_data, coco_dataloader, out_dir)

    # Save the final checkpoint at the end of training
    # save_model_checkpoint(fabric, model, os.path.join(out_dir, "lit-llama-adapter-finetuned.pth"))


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    clip_model,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    coco_dataloader,
    out_dir: str,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """

    step_count = 0
    coco_iter = iter(coco_dataloader)

    for iter_num in range(max_iters):

        if step_count <= warmup_iters:
            # linear warmup
            lr = learning_rate * step_count / warmup_iters
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data)
        # imgs = torch.rand(input_ids.size(0), 3, 256, 256, device=fabric.device)
        # img_features = torch.load("features-0.pt", map_location=fabric.device)["clip_feats"]
        # device = torch.device("cuda", 0)
        # clip_model.to(device)
        
        img, caption = next(coco_iter)
        # img = img.unsqueeze(0).to(fabric.device)
        img = img.to(fabric.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            clip_feats = clip_encode_image(clip_model, img)
        # print(clip_feats.shape)

        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_iters != 0)):
            logits = model(input_ids, clip_feats)
            loss = loss_fn(logits, targets)
            fabric.backward(loss / gradient_accumulation_iters)

        if (iter_num + 1) % gradient_accumulation_iters == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
                
            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_data)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step_count % save_interval == 0:
                pass
                # print(f"Saving adapter weights to {out_dir}")
                # TODO: Provide a function/script to merge the adapter weights with pretrained weights
                # save_model_checkpoint(fabric, model, os.path.join(out_dir, f"iter-{iter_num:06d}.pth"))

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")


def generate_response(model, instruction, input=""):
    tokenizer = Tokenizer("checkpoints/lit-llama/tokenizer.model")
    sample = {"instruction": instruction, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_seq_length,
        max_new_tokens=100,
        temperature=0.8,
    )
    output = tokenizer.decode(output)
    return output # output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    val_loss = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    output = generate_response(model, instruction)
    fabric.print(instruction)
    fabric.print(output)

    model.train()
    return val_loss.item()

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


def load_datasets(data_dir):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


def get_coco_dataloader(clip_transform):
    train_captions = CocoCaptions(
        root="/data/shared/datasets/coco/train2014/",
        annFile="/data/shared/datasets/coco/annotations/captions_train2014.json",
        transform=clip_transform
    )
    val_captions = CocoCaptions(
        root="/data/shared/datasets/coco/val2014/",
        annFile="/data/shared/datasets/coco/annotations/captions_val2014.json",
        transform=clip_transform
    )
    dataset = train_captions #  + val_captions
    print('Number of samples: ', len(dataset))
    return DataLoader(dataset, batch_size=micro_batch_size, shuffle=True, num_workers=0)

# def save_model_checkpoint(fabric, model, file_path):
#     file_path = Path(file_path)

#     if isinstance(fabric.strategy, DeepSpeedStrategy):
#         from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

#         tmp_path = file_path.with_suffix(".tmp")
#         fabric.save(tmp_path, {"model": model})
#         fabric.barrier()
#         if fabric.global_rank == 0:
#             # Create a consolidated checkpoint with the same name next to the deepspeed checkpoint
#             # and only keep the adapter weights
#             state_dict = get_fp32_state_dict_from_zero_checkpoint(tmp_path)
#             state_dict = adapter_v2_state_from_state_dict(state_dict)
#             torch.save(state_dict, file_path)
#             shutil.rmtree(tmp_path)
#     else:
#         state_dict = adapter_v2_state_from_state_dict(model.state_dict())
#         if fabric.global_rank == 0:
#             torch.save(state_dict, file_path)
#         fabric.barrier()



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


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI(main)

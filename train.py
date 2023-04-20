"""
This script is a placeholder for training LLaMA from scratch.
"""

import glob
import os
import time
from functools import partial
from typing import Tuple
from pathlib import Path

import lightning as L
from lightning.fabric.strategies import FSDPStrategy

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import numpy as np

from lit_llama.model import Block, LLaMA, LLaMAConfig
from lit_llama.utils import save_model_checkpoint
import lit_llama.indexed_dataset as indexed_dataset


out_dir = "out/training"
eval_interval = 2000
eval_iters = 200
log_interval = 1
# compilation fails as it does not support torch.complex64 for RoPE
# compile = False

# Hyperparameters
learning_rate = 6e-4
batch_size = 2
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0


def main(
    devices: int = 4,
    train_data_dir: Path = "data/red_pajama",
    val_data_dir: Path = "data/red_pajama_val"
) -> None:
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing=Block)

    fabric = L.Fabric(accelerator="cuda", devices=devices, precision="bf16-mixed", strategy=strategy)
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    config = LLaMAConfig.from_name("7B")

    # TODO: create val data during prepare_redpajama
    train_dataloader, val_dataloader = create_dataloaders(block_size=config.block_size, train_data_dir=train_data_dir, val_data_dir=val_data_dir)

    with fabric.device:
        model = LLaMA(config)

    # if compile:
    #     model = torch.compile(model)

    model = fabric.setup_module(model)

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    optimizer = fabric.setup_optimizers(optimizer)

    train(fabric, model, optimizer, train_dataloader, val_dataloader)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """

    for iter_num, train_data in enumerate(train_dataloader):
        # TODO: add learning rate scheduling

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num > 0 and iter_num % eval_interval == 0:
            val_loss = validate(fabric, model, val_dataloader)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            fabric.print(f"Saving checkpoint to {out_dir}")
            save_model_checkpoint(fabric, model, os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"))

        t0 = time.time()

        input_ids, targets = get_batch(
            fabric,
            train_data,
            block_size=model.config.block_size,  # type: ignore[union-attr,arg-type]
        )

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        fabric.backward(loss)

        # TODO: Gradient clipping
        # if grad_clip != 0.0:
        #     fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")
        iter_num += 1

        if iter_num > max_iters:
            break


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k, val_data in enumerate(val_dataloader):
        input_ids, targets = get_batch(
            fabric,
            val_data,
            block_size=model.config.block_size,  # type: ignore[union-attr,arg-type]
        )
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


# def get_batch(fabric: L.Fabric, data: np.ndarray, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
#     y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
#     x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
#     return x, y


def collate_batch(block_size, batch):
    print("BATCH", [el.shape for el in batch])
    max_len = max(len(el) for el in batch)

    xs = []
    ys = []
    for data in batch:
        i = torch.randint(max(len(data) - block_size - 1, 0), ())
        x = torch.from_numpy((data[i : i + block_size]).astype(np.int64))
        y = torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
        xs.append(x)
        ys.append(y)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    xs = torch.stack([pad_right(x, pad_id=0) for x in xs])
    ys = torch.stack([pad_right(y, pad_id=-1) for y in ys])

    # x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    print("X", xs.shape, xs)
    print("Y", ys.shape, ys)

    return xs, ys


def create_dataloader(block_size: int, data_dir: str, shuffle=False) -> DataLoader:
    datasets = []
    for name in [os.path.splitext(el)[0] for el in glob.glob(os.path.join(data_dir, "*.idx"))]:
        dataset = indexed_dataset.make_dataset(name, impl="infer", skip_warmup=False)
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset.")

    concat_dataset = ConcatDataset(datasets)

    collate_fn = partial(collate_batch, block_size)

    return DataLoader(concat_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def create_dataloaders(block_size: int, train_data_dir: str = "data/red_pajama", val_data_dir: str = "data/red_pajama_val") -> Tuple[DataLoader, DataLoader]:    
    train_dataloader = create_dataloader(block_size=block_size, data_dir=train_data_dir, shuffle=True)
    val_dataloader = create_dataloader(block_size=block_size, data_dir=val_data_dir, shuffle=False)
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI(main)

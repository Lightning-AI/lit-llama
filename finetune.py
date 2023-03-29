import os
import time
from functools import partial
from typing import Tuple

import lightning as L
from lightning.fabric.strategies import FSDPStrategy

import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import numpy as np

from lit_llama.model import Block, LLaMA, LLaMAConfig


out_dir = "out"
eval_interval = 200
eval_iters = 200
log_interval = 1
# compilation fails as it does not support torch.complex64 for RoPE
# compile = False

# Hyperparameters
learning_rate = 3e-4  # alpaca-lora: 3e-4 vs alpaca:  2e-5
batch_size = 2  # 128
max_iters = 50000 * 3 // 4  # roughly 3 epochs across 4 devices
weight_decay = 0.0


# --gradient_accumulation_steps 8 \
# --warmup_ratio 0.03 \

# For shakespeare, choose smaller block size than vanilla LLaMA
block_size = 256


def main() -> None:
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing=Block)

    fabric = L.Fabric(accelerator="cuda", devices=4, precision="bf16-mixed", strategy=strategy)
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets()

    config = LLaMAConfig.from_name("7B")
    config.block_size = block_size

    with fabric.device: # , with_lora(0.0, 1.0, 0.0):
        model = LLaMA(config)

    # if compile:
    #     model = torch.compile(model)

    model = fabric.setup_module(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters, last_epoch=0)

    train(fabric, model, optimizer, scheduler, train_data, val_data)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """

    iter_num = 0

    while True:
        # evaluate the loss on train/val sets and write checkpoints
        if iter_num > 0 and iter_num % eval_interval == 0:
            val_loss = validate(fabric, model, val_data)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            # TODO: Save with Fabric
            # print(f"saving checkpoint to {out_dir}")
            # torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            fabric.barrier()
        

        t0 = time.time()

        input_ids, targets = get_batch(
            fabric,
            train_data,
            # TODO: is the padding id correct?
            # tokenizer says it is -1, but can't be because embedding layer does not support neg idx
        )

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        fabric.backward(loss)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")
        iter_num += 1

        if iter_num > max_iters:
            break


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


def get_batch(fabric: L.Fabric, data: list, pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data), (batch_size,))
    # TODO: don't we need to shift the labels?

    def pad(x):
        # TODO: optimize this to pad to the next multiple of 8 or so?
        n = block_size - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad(data[i]["input_ids"]) for i in ix]).type(torch.int64)
    y = torch.stack([pad(data[i]["labels"]) for i in ix]).type(torch.int64)
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir: str = "data/alpaca"):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()

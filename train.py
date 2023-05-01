import os
import glob
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Optional

import lightning as L
from lightning.fabric.strategies import FSDPStrategy

import torch
from torch.utils.data import DataLoader
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import numpy as np

from lit_llama.model import Block, LLaMA, LLaMAConfig
from lit_llama.packed_dataset import PackedDataset, CombinedDataset
from lit_llama.utils import save_model_checkpoint


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

# For shakespeare, choose smaller block size than vanilla LLaMA
block_size = 1024


def main(
    devices: int = 4,
    train_data_dir: Path = "data/red_pajama",
    val_data_dir: Path = "data/red_pajama_val",
) -> None:
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Block}
    )
    strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy, activation_checkpointing=Block
    )

    fabric = L.Fabric(
        accelerator="cuda", devices=devices, precision="bf16-mixed", strategy=strategy
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    config = LLaMAConfig.from_name("7B")
    config.block_size = block_size
    config.vocab_size = 100  # from prepare_shakespeare.py

    with fabric.device:
        model = LLaMA(config)

    # if compile:
    #     model = torch.compile(model)

    model = fabric.setup_module(model)

    train_dataloader, val_dataloader = create_dataloaders(
        block_size=config.block_size,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=12345,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )
    optimizer = fabric.setup_optimizers(optimizer)

    train(fabric, model, optimizer, train_dataloader, val_dataloader)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """

    for iter_num, train_data in enumerate(train_dataloader):
        # TODO: add learning rate scheduling

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num > 0 and iter_num % eval_interval == 0 and val_dataloader is not None:
            val_loss = validate(fabric, model, val_dataloader)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            fabric.print(f"Saving checkpoint to {out_dir}")
            save_model_checkpoint(
                fabric, model, os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth")
            )

        t0 = time.time()

        input_ids, targets = get_batch(
            fabric, train_data, block_size=model.config.block_size
        )
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )

        fabric.backward(loss)

        # TODO: Gradient clipping
        # if grad_clip != 0.0:
        #     fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            fabric.print(
                f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms"
            )

        if iter_num > max_iters:
            break


@torch.no_grad()
def validate(
    fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader
) -> torch.Tensor:
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
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


DATA_CONFIG = [
    ("arxiv", 1.0),
    ("book", 1.0),
    ("c4", 1.0),
    ("cc", 1.0),
    ("github", 1.0),
    ("stackexchange", 1.0),
    ("wikipedia", 1.0),
]


def create_dataloader(
    block_size: int, data_dir: str, shuffle: bool = True, seed: int = 12345
) -> DataLoader:
    datasets = []
    for prefix, _ in DATA_CONFIG:
        filenames = glob.glob(os.path.join(data_dir, prefix))
        dataset = PackedDataset(
            filenames, n_chunks=10, block_size=block_size, shuffle=shuffle, seed=seed
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in DATA_CONFIG]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets, weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)


def create_dataloaders(
    block_size: int,
    train_data_dir: str = "data/red_pajama",
    val_data_dir: str = "data/red_pajama_val",
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        block_size=effective_block_size,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
    )
    val_dataloader = create_dataloader(
        block_size=effective_block_size, data_dir=val_data_dir, shuffle=False, seed=seed
    ) if val_data_dir else None
    return train_dataloader, val_dataloader


def get_batch(
    fabric: L.Fabric, data: np.ndarray, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy([data[0:block_size]]).astype(np.int64)
    y = torch.from_numpy([data[1 : block_size + 1]]).astype(np.int64)
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()

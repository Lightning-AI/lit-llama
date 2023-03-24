import os
import pickle
import time
from functools import partial

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import models.llama as llama

# TODO: properly structure this file into functions

out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
dataset = "shakespeare"

# TODO: hyperparameters
gradient_accumulation_steps = 1
batch_size = 2
block_size = 1024

# TODO: hparams for LLaMA
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
compile = False  # use PyTorch 2.0 to compile the model to be faster


auto_wrap_policy = partial(
    transformer_auto_wrap_policy, transformer_layer_cls={llama.TransformerBlock}
)
strategy = FSDPStrategy(
    auto_wrap_policy=auto_wrap_policy,
    activation_checkpointing=llama.TransformerBlock,
)

fabric = L.Fabric(
    accelerator="cuda",
    devices=4,
    precision="bf16",
    strategy=strategy,
)
fabric.launch()
fabric.seed_everything(1337 + fabric.global_rank)


if fabric.global_rank == 0:
    os.makedirs(out_dir, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


# poor man's data loader
data_dir = os.path.join("data", dataset)
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )

    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    x, y = x.pin_memory().to(fabric.device, non_blocking=True), y.pin_memory().to(
        fabric.device, non_blocking=True
    )
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")


# init a new model from scratch
print("Initializing a new model from scratch")
llama_config = llama.LLAMA_CONFIG_DICT["7B"]
with fabric.device:
    model = llama.LLaMA(llama_config)


# compile the model
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0

model = fabric.setup_module(model)

# TODO: AdamW from paper
optimizer = torch.optim.Adam(model.parameters())
optimizer = fabric.setup_optimizers(optimizer)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            input_ids, targets = get_batch(split)
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# training loop
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process

while True:
    # TODO: add learning rate scheduling

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num > 0 and iter_num % eval_interval == 0 and fabric.global_rank == 0:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }

            # TODO: Save with Fabric
            # print(f"saving checkpoint to {out_dir}")
            # torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # TODO: efficient gradient accumulation
    # TODO: Should we do gradient accumulation at all?
    for micro_step in range(gradient_accumulation_steps):
        input_ids, targets = get_batch("train")
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        fabric.backward(loss)

    # TODO: Gradient clipping
    # if grad_clip != 0.0:
    #     fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and fabric.global_rank == 0:
        fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

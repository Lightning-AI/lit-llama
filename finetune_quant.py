"""
Instruction-tuning with LoRA on the Alpaca dataset.
"""
import os
import time

import lightning as L
import numpy as np
import torch

from generate import generate
from lit_llama.lora import mark_only_lora_as_trainable, with_lora
# from lit_llama.lora_old import mark_only_lora_as_trainable, with_lora
from lit_llama.model import LLaMA, LLaMAConfig
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
import bitsandbytes as bnb
import wandb

out_dir = "out/lora-quant-orig-dataset"
eval_interval = 4000
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 3e-4
batch_size = 128
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size


max_iters = 100000000
# TODO: Limit to 3 epochs
# max_iters = 50000 * 3 // micro_batch_size
weight_decay = 0.0
block_size = 256

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05

warmup_steps = 100


wandb.init(project="alpaca-lora")

# def convert_weights(state_dict):
#     return {k: v.half() for k, v in state_dict.items()}


def main():
    fabric = L.Fabric(accelerator="cuda", devices=1)
    # fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets()

    config = LLaMAConfig.from_name("7B")
    config.block_size = block_size

    with with_lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
        model = LLaMA(config)
        print(model.transformer.h[0].attn.c_attn.weight.dtype)
        print(type(model.transformer.h[0].attn.c_attn))

    checkpoint = torch.load("checkpoints/lit-llama/7B/state_dict.pth")
    
    # strict=False because missing keys due to lora weights not contained in checkpoint state
    model.load_state_dict(checkpoint, strict=False) 
    mark_only_lora_as_trainable(model)
    
    
    del checkpoint

    # print(model.transformer.h[0].attn.c_attn.weight.dtype)
    # print(type(model.transformer.h[0].attn.c_attn))

    # print(model.transformer.h[0].attn.weight)


    # optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
    # TODO: use this
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model, optimizer = fabric.setup(model, optimizer)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters, last_epoch=-1)


    print(model.transformer.h[0].attn.c_attn.weight.dtype)
    train(fabric, model, optimizer, train_data, val_data)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    best_val_loss = 100000.
    step_count = 0

    for iter_num in range(max_iters):

        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            val_loss = validate(fabric, model, val_data)
            wandb.log({"val_loss": val_loss})
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                print(f"Saving checkpoint to {out_dir}")
                checkpoint = {"model": model, "optimizer": optimizer, "iter": iter, "val_loss": val_loss}
                fabric.save(os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pt"), checkpoint)
                best_val_loss = val_loss
            fabric.barrier()

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data)

        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        fabric.backward(loss)

        fabric.clip_gradients(model, optimizer, clip_val=1.0)

        if (iter_num + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            step_count += 1

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            wandb.log({"train_loss": loss.item()})
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")


def generate_response(model, instruction):
    tokenizer = Tokenizer("checkpoints/lit-llama/tokenizer.model")
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=True)
    encoded = encoded[None, :]  # add batch dimension
    encoded = encoded.to(model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=block_size,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output[0].cpu())
    return output # output.split("### Response:")[1].strip()


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

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    
    output = generate_response(model, instruction)
    fabric.print(instruction)
    fabric.print(output)

    columns = ["instruction", "output"]
    my_data = [[instruction, output]]
    metrics = {"examples": wandb.Table(columns=columns, data=my_data)}
    wandb.log(metrics)

    model.train()
    return out.item()


def get_batch(fabric: L.Fabric, data: list, pad_id: int = 0):
    ix = torch.randint(len(data), (micro_batch_size,))

    def pad(x):
        # TODO: optimize this to pad to the next multiple of 8 or so?
        n = block_size - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    def shift_right(x):
        return x[1:]

    x = torch.stack([pad(torch.tensor(data[i]["input_ids"])) for i in ix]).type(torch.int64)
    y = torch.stack([pad(shift_right(torch.tensor(data[i]["labels"]))) for i in ix]).type(torch.int64)
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir: str = "data/alpaca"):
    train_data = torch.load(os.path.join(data_dir, "train_orig.pt"))
    val_data = torch.load(os.path.join(data_dir, "test_orig.pt"))
    return train_data, val_data


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()

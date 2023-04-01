"""
Instruction-tuning with LLaMA-Adapter on the Alpaca dataset.
"""
import os
import time

import lightning as L
import numpy as np
import torch

from generate import generate
from lit_llama.model import LLaMA, LLaMAConfig
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt

import wandb

out_dir = "out/adapter-4gpu-pad-right"
eval_interval = 40
save_interval = 200
eval_iters = 100
log_interval = 1

# Hyperparameters
learning_rate = 9e-3
batch_size = 64
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size
epoch_size = 50000  # train dataset size
num_epochs = 100
max_iters = epoch_size * num_epochs // micro_batch_size  # 5 epochs
weight_decay = 0.02
block_size = 256
warmup_steps = epoch_size * 2 // micro_batch_size  # 2 epochs


def main():
    wandb.init(project="llama-adapter")

    fabric = L.Fabric(accelerator="cuda", devices=4, strategy="ddp")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets()

    config = LLaMAConfig.from_name("7B")
    config.block_size = block_size

    with fabric.device:
        model = LLaMA(config)
    
    checkpoint = torch.load("checkpoints/lit-llama/7B/state_dict.pth")
    # strict=False because missing keys due to adapter weights not containted in state dict
    model.load_state_dict(checkpoint, strict=False)

    # mark only the adapter weights as trainable
    for name, param in model.named_parameters():
        param.requires_grad = "adapter_wte" in name or "gating_factor" in name

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
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
    step_count = 0

    for iter_num in range(max_iters):

        if step_count <= warmup_steps:
            # linear warmup
            lr = learning_rate * step_count / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_steps != 0)):
            fabric.backward(loss)

        fabric.clip_gradients(model, optimizer, clip_val=1.0)

        if (iter_num + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
                
            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_data)
                wandb.log({"val_loss": val_loss}, commit=False)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step_count % save_interval == 0:
                pass
                # print(f"Saving LoRA weights to {out_dir}")
                
                # only save the adapter weights
                # TODO: make this a function
                checkpoint = {name: param for name, param in model.named_parameters() if "adapter_wte" in name or "gating_factor" in name}

                # TODO: Provide a function/script to merge the adapter weights with pretrained weights
                # checkpoint = adapter_state_dict(model)
                fabric.save(os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pt"), checkpoint)

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            wandb.log({"train_loss": loss.item(), "step": step_count, "epoch_pct": iter_num * micro_batch_size / 50000})
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


example_outputs = []


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
    out = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    
    output = generate_response(model, instruction)
    fabric.print(instruction)
    fabric.print(output)

    columns = ["output"]
    example_outputs.append([output])
    metrics = {"examples": wandb.Table(columns=columns, data=example_outputs)}
    wandb.log(metrics)

    model.train()
    return out.item()

def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss
    

def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [torch.tensor(data[i]["input_ids"], dtype=torch.int64) for i in ix]
    labels = [torch.tensor(data[i]["labels"], dtype=torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir: str = "data/alpaca"):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()

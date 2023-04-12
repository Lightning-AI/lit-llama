"""
Instruction-tuning with LLaMA-Adapter on the Alpaca dataset following the paper

LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention
https://arxiv.org/abs/2303.16199

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", install
the PyTorch nightly version for a fix (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import os
import time

import lightning as L
import numpy as np
import torch

from generate import generate
from lit_llama.adapter import LLaMA, LLaMAConfig, mark_only_adapter_as_trainable, adapter_state_dict, Block
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
from lightning.fabric.strategies import DeepSpeedStrategy

import wandb

out_dir = "out/adapter/full-training-ds"
eval_interval = 600
save_interval = 1000
eval_iters = 100
log_interval = 1
devices = 6

# Hyperparameters
learning_rate = 9e-3
batch_size = 64 / devices
micro_batch_size = 8
gradient_accumulation_steps = batch_size // micro_batch_size
epoch_size = 50000  # train dataset size
num_epochs = 5
max_iters = num_epochs * epoch_size // devices  # 5 epochs
weight_decay = 0.02
block_size = 512
warmup_steps = epoch_size * 2 // micro_batch_size // devices  # 2 epochs

ds_config = {
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "zero_optimization": {"stage": 2},
}


def main():
    fabric = L.Fabric(accelerator="cuda", devices=devices, strategy=DeepSpeedStrategy(config=ds_config), precision="bf16")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.is_global_zero:
        wandb.init(project="llama-adapter", notes="deepspeed with bfloat16 new logging and hparams")

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets()

    config = LLaMAConfig()
    config.block_size = block_size

    checkpoint = torch.load("checkpoints/lit-llama/7B/state_dict.pth")

    with fabric.device:
        model = LLaMA(config).bfloat16()
        # strict=False because missing keys due to adapter weights not containted in state dict
        model.load_state_dict(checkpoint, strict=False)
    
    mark_only_adapter_as_trainable(model)

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, val_data)

    # save at end of training
    checkpoint = {"model": model, "optimizer": optimizer}
    fabric.save(os.path.join(out_dir, f"alpaca-adapter-finetuned.pt"), checkpoint)


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

    # initial validation
    val_loss = validate(fabric, model, val_data)

    # sanity check that saving works
    checkpoint = {"model": model, "optimizer": optimizer}
    fabric.save(os.path.join(out_dir, f"sanity.pt"), checkpoint)

    # training
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
            fabric.backward(loss / gradient_accumulation_steps)

        # fabric.clip_gradients(model, optimizer, clip_val=1.0)

        if (iter_num + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1
                
            if step_count % eval_interval == 0:
                val_loss = validate(fabric, model, val_data)
                fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            # if step_count % save_interval == 0:
            #     pass
            #     print(f"Saving adapter weights to {out_dir}")
                
                
                # only save the adapter weights for smaller checkpoint files
                # checkpoint = adapter_state_dict(model)
                # TODO: Provide a function/script to merge the adapter weights with pretrained weights
                # if fabric.is_global_zero:
                    # torch.save(checkpoint, os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pt"))
                # fabric.barrier()

        dt = time.time() - t0
        if iter_num % log_interval == 0:
            if fabric.is_global_zero:
                wandb.log({
                    "train_loss": loss.item(), 
                    "train_loss_n": loss.item() / gradient_accumulation_steps,
                    "train_ppl": torch.exp(loss).item(),
                    "train_ppl_n": torch.exp(loss / gradient_accumulation_steps).item(),
                    "step": step_count, 
                    # "epoch_pct": iter_num * micro_batch_size / 50000
                })
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")

example_outputs = []


def generate_response(model, instruction, input=""):
    tokenizer = Tokenizer("checkpoints/lit-llama/tokenizer.model")
    sample = {"instruction": instruction, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=True)
    encoded = encoded[None, :]  # add batch dimension
    encoded = encoded.to(model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=block_size,
        max_new_tokens=100,
        temperature=0.1,
    )
    output = tokenizer.decode(output[0].cpu())
    return output # output.split("### Response:")[1].strip()

def generate_response_batch(model, input_ids):
    tokenizer = Tokenizer("checkpoints/lit-llama/tokenizer.model")

    output = [generate(
        model,
        idx=idx[None, :],
        max_seq_length=block_size,
        max_new_tokens=100,
        temperature=0.8,
    ) for idx in input_ids]
    output = [tokenizer.decode(output[k][0].cpu()) for k in range(len(output))]
    input = [tokenizer.decode(input_ids[k].cpu()) for k in range(len(input_ids))]
    return input, output # output.split("### Response:")[1].strip()


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

    # get an additional batch for logging responses
    input_ids = get_test_samples(fabric, val_data)
    text_input, text_output = generate_response_batch(model, input_ids)

    # # produce an example:
    # instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    
    # output = generate_response(model, instruction)
    # fabric.print(instruction)
    # fabric.print(output)
    
    if fabric.is_global_zero:
        columns = ["input", "output"]
        example_outputs.extend([[txt, txt_out] for txt, txt_out in zip(text_input, text_output)])
        metrics = {"examples": wandb.Table(columns=columns, data=example_outputs)}
        wandb.log(metrics, commit=False)

    val_ppl = torch.exp(val_loss)
    if fabric.is_global_zero:
        wandb.log({"val_loss": val_loss, "val_ppl": val_ppl}, commit=False)

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


def get_test_samples(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))
    input_ids = [data[i]["input_ids_no_response"].type(torch.int64) for i in ix]
    input_ids = fabric.to_device(input_ids)
    return input_ids


def load_datasets(data_dir: str = "data/alpaca"):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()

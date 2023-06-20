"""
This script is a placeholder for training LLaMA from scratch.
Currently, it just trains on the Shakespeare dataset.
"""
import datetime
from pathlib import Path
import sys
import os
import time
from functools import partial
from typing import Tuple

import lightning as L
from lightning.fabric.strategies import FSDPStrategy

import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import numpy as np

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama.model import Block, LLaMA, LLaMAConfig
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


def print_memory(label: str, fabric: L.Fabric):
    memory_use = fabric.all_gather(torch.Tensor([
        torch.cuda.memory_allocated(),
        torch.cuda.max_memory_allocated(),
        torch.cuda.memory_reserved(),
        torch.cuda.max_memory_reserved(),
    ]) / 1024 ** 3).cpu().t().numpy()

    lines = [f"{label:<20} allocated   (max)  |  reserved   (max)"]
    for rank, (allocated, max_allocated, reserved, max_reserved) in enumerate(zip(*memory_use)):
        lines.append(f"  rank:{rank:<13} {allocated:>8.1f} {max_allocated:>8.1f}  : {reserved:>8.1f} {max_reserved:8.1f}")
    fabric.print("\n".join(lines), "\n")


class MemoryProfileTimeline_WithReserved(torch.profiler.profiler.MemoryProfileTimeline):

    def __init__(self, profile):
        super().__init__(profile._memory_profile())
        self.event_tree = profile.profiler.kineto_results.experimental_event_tree()

    # Modified: https://github.com/pytorch/pytorch/blob/a60f6dbe69b0cae2e27ac279cdf399c6c16650af/torch/profiler/_memory_profiler.py#L1049-L1101
    def export_memory_timeline_html(self, path, device, figsize=(20, 12), title=None) -> None:
        """Exports the memory timeline as an HTML file which contains
        the memory timeline plot embedded as a PNG file."""
        from torch.profiler._memory_profiler import _CATEGORY_TO_COLORS, _CATEGORY_TO_INDEX

        # Check if user has matplotlib installed, return gracefully if not.
        import importlib.util
        matplotlib_spec = importlib.util.find_spec("matplotlib")
        if matplotlib_spec is None:
            print("export_memory_timeline_html failed because matplotlib was not found.")
            return

        import matplotlib.pyplot as plt
        import numpy as np
        from base64 import b64encode
        from tempfile import NamedTemporaryFile
        from os import remove

        mt = self._coalesce_timeline(device)
        times, sizes = np.array(mt[0]), np.array(mt[1])
        stacked = np.cumsum(sizes, axis=1) / 1024**3

        # Plot memory timeline as stacked data
        fig = plt.figure(figsize=figsize, dpi=80)
        axes = fig.gca()
        for category, color in _CATEGORY_TO_COLORS.items():
            i = _CATEGORY_TO_INDEX[category]
            axes.fill_between(
                times / 1e3, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7
            )

        from torch._C._profiler import _EventType
        from torch.profiler._utils import traverse_dfs
        allocations = [e for e in traverse_dfs(self.event_tree) if e.tag == _EventType.Allocation and str(e.extra_fields.device) == device]
        allocations.sort(key=lambda e: e.start_time_ns)
        axes.plot(
            [e.start_time_ns / 1e3 for e in allocations],
            [e.extra_fields.total_reserved / 1024 ** 3 for e in allocations],
            color="darkslategrey", linewidth=2,
        )

        fig.legend(["Unknown" if i is None else i.name for i in _CATEGORY_TO_COLORS] + ["RESERVED"])
        axes.set_xlabel("Time (us)")
        axes.set_ylabel("Memory (GB)")
        title = "\n\n".join(
            ([title] if title else []) + [f"Max: {stacked[:, -1].max():.2f} GB"]
        )
        axes.set_title(title)

        # Embed the memory timeline image into the HTML file
        tmpfile = NamedTemporaryFile('wb', suffix='.png', delete=False)
        tmpfile.close()
        fig.savefig(tmpfile.name, format='png')

        with open(tmpfile.name, 'rb') as tmp:
            encoded = b64encode(tmp.read()).decode('utf-8')
            html = """<html>
    <head><meta charset="utf-8" /><title>GPU Memory Timeline HTML</title></head>
    <body>
    <img src=\'data:image/png;base64,{}\'>
    </body>
    </html>""".format(encoded)

            with open(path, 'w') as f:
                f.write(html)
        remove(tmpfile.name)


def main() -> None:
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})

    from torch.distributed.fsdp import BackwardPrefetch
    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing=Block, limit_all_gathers=True, backward_prefetch=BackwardPrefetch.BACKWARD_POST)

    fabric = L.Fabric(accelerator="cuda", devices=4, precision="bf16-mixed", strategy=strategy)
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    train_data, val_data = load_datasets()

    config = LLaMAConfig.from_name("7B")
    config.block_size = block_size
    config.vocab_size = 100  # from prepare_shakespeare.py

    with fabric.device:
        model = LLaMA(config)

    # if compile:
    #     model = torch.compile(model)

    model = fabric.setup_module(model)
    train(fabric, model, train_data, val_data)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    train_data: np.ndarray,
    val_data: np.ndarray,
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    print_memory("Begin", fabric)
    iter_num = 0

    optimizer_kwargs = dict(lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False)

    if os.getenv("OPTIMIZER_IN_BACKWARD") not in (None, "", "0", "False"):
        from lightning.fabric.strategies.fsdp import fsdp_overlap_step_with_backward

        optimizers = [torch.optim.AdamW([p], **optimizer_kwargs) for p in model.parameters()]
        optimizers = fabric.setup_optimizers(*optimizers)

        def backward_and_step(loss):
            with fsdp_overlap_step_with_backward(optimizers, model):
                fabric.backward(loss)

    else:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        optimizer = fabric.setup_optimizers(optimizer)

        def backward_and_step(loss):
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()


    def step():
        input_ids, targets = get_batch(
            fabric,
            train_data,
            block_size=model.config.block_size,  # type: ignore[union-attr,arg-type]
        )
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        backward_and_step(loss)
        return loss

    times = []
    for _ in range(10):
        # TODO: add learning rate scheduling

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num > 0 and iter_num % eval_interval == 0:
            val_loss = validate(fabric, model, val_data)
            fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            fabric.print(f"Saving checkpoint to {out_dir}")
            save_model_checkpoint(fabric, model, os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"))

        t0 = time.time()
        loss = step()
        dt = time.time() - t0
        times.append(dt)
        if iter_num % log_interval == 0:
            fabric.print(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")
            print_memory("", fabric)
        iter_num += 1

        if iter_num > max_iters:
            break

    with torch.profiler.profile(with_stack=True, record_shapes=True, profile_memory=True) as p:
        _ = step()

    times = times[1:]  # First step is generally slower due to initialization
    mean = torch.mean(torch.tensor(times)).item()
    median = torch.median(torch.tensor(times)).item()
    fabric.print(f"Times:\n  mean:   {mean}\n  median: {median}\n  {times}")
    if fabric.global_rank == 0:
        trace_dir = Path(__file__).parent.parent.joinpath("traces")
        trace_dir.mkdir(exist_ok=True)
        now = datetime.datetime.now().strftime("%Y_%m_%d:%H.%M.%S")
        MemoryProfileTimeline_WithReserved(p).export_memory_timeline_html(
            path=str(trace_dir.joinpath(f"LLaMA_{now}_with_reserved.html")),
            device=str(fabric.device),
        )
        p.export_memory_timeline(
            path=str(trace_dir.joinpath(f"LLaMA_{now}.html")),
            device=str(fabric.device),
        )
        p.export_chrome_trace(str(trace_dir.joinpath(f"LLaMA_{now}.pt.trace.json.gz")))


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
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


def get_batch(fabric: L.Fabric, data: np.ndarray, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir: str = "data/shakespeare") -> Tuple[np.ndarray, np.ndarray]:
    train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return train_data, val_data


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()

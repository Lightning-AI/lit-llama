import sys
import time
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import torch.nn as nn

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama import LLaMA
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from lit_llama.lora import lora

def del_lora_state_dict(model: nn.Module):
    base_model_dict = model.state_dict()
    key_to_delete = [k for k in base_model_dict if "lora_" in k]
    for del_key in key_to_delete:
        del base_model_dict[del_key]
    return base_model_dict


def lora_model_lookup(checkpoint: dict) -> int:
    """Returns the LoRA rank from the adapter checkpoint.

    """
    return checkpoint["transformer.h.0.attn.c_attn.lora_B"].shape[1]
     

def main(
    accelerator: str = "auto",
    lora_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    dtype: str = "bfloat16",
) -> None:
    """Merges lora weights to base model.

    Args:
        accelerator: The hardware to run on. Possible choices are:
            ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        lora_path: Path to the checkpoint with trained LoRA weights, which are the output of
            `finetune_lora.py`.
        checkpoint_path: The checkpoint path to load.
        dtype: `torch.dtype` to work with
    """
    if not lora_path:
        lora_path = Path("out/lora/alpaca/lit-llama-lora-finetuned.pth")
    if not checkpoint_path:
        checkpoint_path = Path(f"./checkpoints/lit-llama/7B/lit-llama.pth")

    assert lora_path.is_file()
    assert checkpoint_path.is_file()

    fabric = L.Fabric(accelerator=accelerator, devices=1)

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()

    with (lazy_load(checkpoint_path) as pretrained_checkpoint,
          lazy_load(lora_path) as lora_checkpoint):
        name = llama_model_lookup(pretrained_checkpoint)
        rank = lora_model_lookup(lora_checkpoint)

        with EmptyInitOnDevice(
                device=fabric.device, dtype=dtype
        ), lora(r=rank, alpha=16, dropout=0.05, enabled=True):
            model = LLaMA.from_name(name)

            # 1. Load the pretrained weights
            model.load_state_dict(pretrained_checkpoint, strict=False)
            # 2. Load the fine-tuned lora weights
            model.load_state_dict(lora_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    base_model_dict = del_lora_state_dict(model)
    save_path = lora_path.with_stem(f"{lora_path.stem}-lora-merged-weights")
    print("Saving LoRA to base model weights ...")
    torch.save(base_model_dict, save_path)
    print(f"Model saved at {save_path}")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
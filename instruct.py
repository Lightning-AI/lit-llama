import sys
import time
from pathlib import Path
from typing import Optional

import lightning as L
import torch

from lit_llama import Tokenizer
from lit_llama.adapter import LLaMA, LLaMAConfig
from lit_llama.utils import EmptyInitOnDevice
from generate import generate
from scripts.prepare_alpaca import generate_prompt


def main(
    prompt: str = "Hello, my name is",
    *,
    input: str = "",
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    # compilation fails as it does not support torch.complex64 for RoPE
    # compile: bool = False,
    accelerator: str = "auto",
    checkpoint_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    model_size: str = "7B",
    dtype: Optional[str] = None,
    quantize: Optional[str] = None,
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        # compile: Whether to compile the model.
        accelerator: The hardware to run on. Possible choices are:
            ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
    """
    if not checkpoint_path:
        checkpoint_path = Path(f"./checkpoints/lit-llama/{model_size}/state_dict.pth")
    if not tokenizer_path:
        tokenizer_path = Path("./checkpoints/lit-llama/tokenizer.model")
    assert checkpoint_path.is_file()
    assert tokenizer_path.is_file()

    fabric = L.Fabric(accelerator=accelerator, devices=1)

    if dtype is not None:
        dt = getattr(torch, dtype, None)
        if not isinstance(dt, torch.dtype):
            raise ValueError(f"{dtype} is not a valid dtype.")
        dtype = dt

    with EmptyInitOnDevice(
        device=fabric.device, dtype=dtype, quantization_mode=quantize
    ):
        print("Loading model ...", file=sys.stderr)
        t0 = time.time()
        model = LLaMA(LLaMAConfig())
        pretrained_checkpoint = torch.load(f"./checkpoints/lit-llama/{model_size}/state_dict.pth")
        model.load_state_dict(pretrained_checkpoint, strict=False)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()

    # if compile:
    #     model = torch.compile(model)

    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)
    encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False, device=fabric.device)
    encoded_prompt = encoded_prompt[None, :]  # add batch dimension

    t0 = time.perf_counter()

    sample = {"instruction": prompt, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False)
    encoded = encoded[None, :]  # add batch dimension
    encoded = encoded.to(model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_new_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    output = tokenizer.decode(output[0].cpu())
    output.split("### Response:")[1].strip()

    # Truncate the output at the EOS token (if present)
    # eos_string = tokenizer.decode(torch.tensor(tokenizer.eos_id, dtype=torch.int64))
    # output = output.split(eos_string)[0]

    print(output)
    t = time.perf_counter() - t0

    print(f"\n\nTime for inference: {t:.02f} sec total, {max_new_tokens / t:.02f} tokens/sec", file=sys.stderr)
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)

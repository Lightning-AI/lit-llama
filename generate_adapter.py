import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

from generate import generate
from lit_llama import Tokenizer
from lit_llama.adapter import LLaMA, LLaMAConfig
from lit_llama.utils import EmptyInitOnDevice
from scripts.prepare_alpaca import generate_prompt


def main(
    prompt: str = "What food do lamas eat?",
    *,
    input: str = "",
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    accelerator: str = "auto",
    pretrained_path: Optional[Path] = None,
    adapter_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    model_size: str = "7B",
    dtype: Optional[str] = None,
    quantize: Optional[str] = None,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script is mean to work with checkpoints from an instruction-tuned model such as LLaMA-Adapter.
    See `finetune_adapter.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        accelerator: The hardware to run on. Possible choices are:
            ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
        adapter_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune_adapter.py`.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
    """
    if not pretrained_path:
        pretrained_path = Path(f"./checkpoints/lit-llama/{model_size}/state_dict.pth")
    if not tokenizer_path:
        tokenizer_path = Path("./checkpoints/lit-llama/tokenizer.model")
    assert pretrained_path.is_file()
    assert adapter_path.is_file()
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
        pretrained_checkpoint = torch.load(pretrained_path)
        model.load_state_dict(pretrained_checkpoint, strict=False)
        adapter_checkpoint = torch.load(adapter_path)
        model.load_state_dict(adapter_checkpoint, strict=False)
        print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
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
    output = truncate_output_to_eos(output[0].cpu(), tokenizer.eos_id)
    
    output = tokenizer.decode(output)
    output = output.split("### Response:")[1].strip()

    print(output)
    t = time.perf_counter() - t0

    print(f"\n\nTime for inference: {t:.02f} sec total, {max_new_tokens / t:.02f} tokens/sec", file=sys.stderr)
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


def truncate_output_to_eos(output, eos_id):
    # The end of the response is where the model generates the EOS token
    # TODO: Make this more efficient, terminate generation early
    try:
        eos_pos = output.tolist().index(eos_id)
    except ValueError:
        eos_pos = -1

    output = output[:eos_pos]
    return output


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)

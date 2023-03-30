import sys
import time
from pathlib import Path
from typing import Optional

import lightning as L
import torch

from lit_llama import LLaMA, Tokenizer, as_8_bit_quantized
from generate import generate
from finetune import generate_prompt


def main(
    prompt: str = "Hello, my name is",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: int = 200,
    temperature: float = 0.8,
    # compilation fails as it does not support torch.complex64 for RoPE
    # compile: bool = False,
    accelerator: str = "auto",
    checkpoint_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    model_size: str = "7B",
    quantize: bool = False,
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        # compile: Whether to compile the model.
        accelerator: The hardware to run on. Possible choices are:
            ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model using the `LLM.int8()` method
    """
    if not checkpoint_path:
        checkpoint_path = Path(f"./checkpoints/lit-llama/{model_size}/state_dict.pth")
    if not tokenizer_path:
        tokenizer_path = Path("./checkpoints/lit-llama/tokenizer.model")
    assert checkpoint_path.is_file()
    assert tokenizer_path.is_file()

    fabric = L.Fabric(accelerator=accelerator, devices=1)

    with as_8_bit_quantized(fabric.device, enabled=quantize):
        print("Loading model ...", file=sys.stderr)
        t0 = time.time()
        model = LLaMA.from_name(model_size)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"], strict=False)
        print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()

    # if compile:
    #     model = torch.compile(model)

    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": prompt, "input": ""}
    prompt = generate_prompt(sample)
    encoded_prompt = tokenizer.encode(prompt, bos=True, eos=True, device=fabric.device)
    encoded_prompt = encoded_prompt[None, :]  # add batch dimension

    L.seed_everything(1234)
    t0 = time.perf_counter()

    for _ in range(num_samples):
        y = generate(
            model,
            encoded_prompt,
            max_new_tokens,
            model.config.block_size,  # type: ignore[union-attr,arg-type]
            temperature=temperature,
            top_k=top_k,
        )[0]  # unpack batch dimension
        print(tokenizer.decode(y))

    t = time.perf_counter() - t0
    print(f"\n\nTime for inference: {t:.02f} sec total, {num_samples * max_new_tokens / t:.02f} tokens/sec", file=sys.stderr)
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)

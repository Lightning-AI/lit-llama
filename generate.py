import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import EmptyInitOnDevice


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = idx.size(0)
    T_new = T + max_new_tokens
    empty = torch.empty(T_new, dtype=idx.dtype, device=idx.device)
    empty[:T] = idx
    idx = empty

    # generate max_new_tokens tokens
    for t in range(T, T_new):
        # ignore the not-filled-yet tokens
        idx_cond = idx[:t]
        # if the sequence context is growing too long we must crop it at max_seq_length
        idx_cond = idx_cond if T <= max_seq_length else idx_cond[-max_seq_length:]

        # forward
        logits = model(idx_cond)
        logits = logits[-1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[[-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # concatenate the new generation
        idx[t] = idx_next

    return idx


def main(
    prompt: str = "Hello, my name is",
    *,
    num_samples: int = 1,
    max_new_tokens: int = 50,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    model_size: str = "7B",
    quantize: Optional[str] = None,
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        model_size: The model size to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
    """
    if not checkpoint_path:
        checkpoint_path = Path(f"./checkpoints/lit-llama/{model_size}/lit-llama.pth")
    if not tokenizer_path:
        tokenizer_path = Path("./checkpoints/lit-llama/tokenizer.model")
    assert checkpoint_path.is_file()
    assert tokenizer_path.is_file()

    fabric = L.Fabric(accelerator="cuda", devices=1)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    with EmptyInitOnDevice(
        device=fabric.device, dtype=dtype, quantization_mode=quantize
    ):
        print("Loading model ...", file=sys.stderr)
        t0 = time.time()
        model = LLaMA.from_name(model_size)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)
    encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False, device=fabric.device)

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
        )
        print(tokenizer.decode(y))

    t = time.perf_counter() - t0
    print(f"\n\nTime for inference: {t:.02f} sec total, {num_samples * max_new_tokens / t:.02f} tokens/sec", file=sys.stderr)
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    warnings.filterwarnings(
        # Triggered in bitsandbytes/autograd/_functions.py:298
        "ignore", 
        message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
    )
    CLI(main)

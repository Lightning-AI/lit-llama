import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama.adapter_v2 import (
    LLaMA, LLaMAConfig,
    adapter_state_from_state_dict,
    mark_instruction_adapter_trainable,
    mark_visual_adapter_trainable,
)
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
from dataset import get_dataloader
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from PIL import Image
import clip
from finetune.adapter_v2_llava import clip_encode_image


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    image_features,
    max_new_tokens: int,
    max_seq_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
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
        eos_id: If specified, stop generating any more token once the <eos> token is triggered
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
        idx_cond = idx_cond if t <= max_seq_length else idx_cond[-max_seq_length:]

        # forward
        logits = model(idx_cond.view(1, -1), image_features)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[[-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # concatenate the new generation
        # https://github.com/pytorch/pytorch/issues/101936
        idx[t] = idx_next.item() if idx.device.type == "mps" else idx_next

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:t + 1]  # include the EOS token

    return idx


def main(
    prompt: str = "What food do lamas eat?",
    image: str = "",
    adapter_path: Path = Path("out/adapter_v2/llava/iter-012799.pth"),
    pretrained_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    quantize: Optional[str] = None,
    max_new_tokens: int = 256,
    max_seq_length: int = 4096,
    top_k: int = 200,
    temperature: float = 0.8,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned LLaMA-Adapter model.
    See `finetune_adapter.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        adapter_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune_adapter.py`.
        input: Optional input (Alpaca style).
        pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
    """
    assert adapter_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    L.seed_everything(0)

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with fabric.init_module(empty_init=True):
            model = LLaMA.from_name(name)

        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned adapter weights
        model.load_state_dict(adapter_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": prompt, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
    prompt_length = encoded.size(0)

    clip_model, clip_transform = clip.load("ViT-L/14")
    clip_model = fabric.to_device(clip_model)
    image = Image.open(image).convert('RGB')
    image = clip_transform(image)
    image = fabric.to_device(image).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast(): 
        image_features = clip_encode_image(clip_model, image)

    t0 = time.perf_counter()
    y = generate(model, encoded, image_features, max_new_tokens=max_new_tokens, max_seq_length=max_seq_length, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
    t = time.perf_counter() - t0

    output = tokenizer.decode(y)
    output = output.split("### Response:")[1].strip()
    print(output)

    tokens_generated = y.size(0) - prompt_length
    print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
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
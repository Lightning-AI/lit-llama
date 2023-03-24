# adapted from karpathy/minGPT
import torch
import models.llama as llama
import lightning as L
import torch.nn as nn
import os

import bitsandbytes as bnb
os.environ["BITSANDBYTES_NOWELCOME"] = "1"


def quantize(model, threshold=6.0, skip=()):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name not in skip:
            model._modules[name] = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias,
                has_fp16_weights=False,
                threshold=threshold,
            )

        if module.children():
            quantize(module, threshold=threshold, skip=skip)
    return model

def generate(
    prompt: str = "Hello, my name is",
    *,
    num_samples: int = 1,
    steps: int = 20,
    top_k: int = 200,
    temperature: float = 0.8,
    compile: bool = False,
    accelerator: str = "auto",
    precision: str = "32-true"
):
    """
    Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        steps: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        compile: Whether to compile the model.
        accelerator: The hardware to run on. Possible choices are:
            ``"cpu"``, ``"cuda"``, ``"mps"``, ``"gpu"``, ``"tpu"``, ``"auto"``.
        precision: Double precision (``"64"``), full precision (``"32"``), half precision AMP (``"16-mixed"``),
            or bfloat16 precision AMP (``"bf16-mixed"``).
    """
    L.seed_everything(1234)

    fabric = L.Fabric(accelerator=accelerator, precision=precision, devices=1)

    checkpoint = torch.load("/srv/data/checkpoints/llama/converted_meta/7B/state_dict.pt")
    llama_config = llama.LLAMA_CONFIG_DICT["7B"]


    model = llama.LLaMA(llama_config)
    model = quantize(model, skip="output")
    model.load_state_dict(checkpoint)

    model.eval()
    model = model.to(fabric.device)

    # TODO: fix this later in the model (buffer)
    model.cos_cached = model.cos_cached.to(fabric.device)
    model.sin_cached = model.sin_cached.to(fabric.device)
    
    if compile:
        model = torch.compile(model)
    # model = fabric.setup_module(model)


    # print(model.output.weight.dtype)
    # print(model.output.weight.device)

    tokenizer = llama.Tokenizer("/srv/data/checkpoints/llama/converted_meta/tokenizer.model")
    encoded_prompt = tokenizer.encode(prompt, bos=True, eos=False).to(fabric.device)
    encoded_prompt = encoded_prompt[None, :]
    for _ in range(num_samples):
        y = model.generate(encoded_prompt, steps, temperature=temperature, top_k=top_k)
        print(tokenizer.decode(y[0]))

    print(torch.cuda.memory_summary())
    print(torch.cuda.max_memory_allocated() // 1e9)
    print(torch.cuda.max_memory_reserved() // 1e9)


if __name__ == "__main__":
    # from jsonargparse import CLI

    # CLI()
    generate()
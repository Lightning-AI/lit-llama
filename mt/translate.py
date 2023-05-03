
import sys
import time
import pandas as pd
from pathlib import Path
from typing import Optional
from jsonargparse import CLI
import lightning as L
import torch

from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
# from generate.py script
from generate import generate


def main_translate(
    input_csv_file: str,
    output_csv_file: str,
    max_new_tokens: int = 128,
    top_k: int = 200,
    temperature: float = 0,
    checkpoint_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
):
    ##### COPIED FROM main(...) IN generate.py #####
    if not checkpoint_path:
        checkpoint_path = Path(f"./checkpoints/lit-llama/7B/lit-llama.pth")
    if not tokenizer_path:
        tokenizer_path = Path("./checkpoints/lit-llama/tokenizer.model")
    assert checkpoint_path.is_file(), checkpoint_path
    assert tokenizer_path.is_file(), tokenizer_path

    fabric = L.Fabric(devices=1)
    dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    checkpoint = lazy_load(checkpoint_path)
    name = llama_model_lookup(checkpoint)

    with EmptyInitOnDevice(
        device=fabric.device, dtype=dtype, quantization_mode=quantize
    ):
        model = LLaMA.from_name(name)

    model.load_state_dict(checkpoint)
    print(
        f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)
    L.seed_everything(1234)
    #################################################
    # load the csv with the prompts to generate on
    df = pd.read_csv(input_csv_file)
    # generate output for each prompt
    for i, row in df.iterrows():
        prompt = row["input"]
        encoded_prompt = tokenizer.encode(
            prompt, bos=True, eos=False, device=fabric.device)
        y = generate(
            model,
            encoded_prompt,
            max_new_tokens,
            model.config.block_size,  # type: ignore[union-attr,arg-type]
            temperature=temperature,
            top_k=top_k,
        )

        df.loc[i, "mt"] = y
    df.to_csv(output_csv_file, index=False)


if __name__ == "__main__":
    CLI(main_translate)

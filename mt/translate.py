import re
import sys
import time
import tqdm
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
        input_csv_file: str = "./mt/data/wmt19_en-de.csv" ,
        output_csv_file: str = "./mt/data/wmt19_en-de_translation.csv",
        max_new_tokens: int = 128,
        top_k: int = 200,
        temperature: float = 0,
        checkpoint_path: Optional[Path] = None,
        tokenizer_path: Optional[Path] = None,
        quantize: Optional[str] = None,
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
    df = pd.read_csv(input_csv_file)[:10]
    # generate output for each prompt
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Translating..."):
        prompt = row["input"]
        encoded_prompt = tokenizer.encode(
            prompt, bos=True, eos=False, device=fabric.device)
        y = generate(
            model = model,
            idx = encoded_prompt,
            max_new_tokens=max_new_tokens,
            max_seq_length=model.config.block_size,  # type: ignore[union-attr,arg-type]
            temperature=temperature,
            top_k=top_k,
            eos_id = None
            # eos_id=13 #int(tokenizer.encode("\n")[-1]) # 13 is the token id for \n
        )
        # extract just the translation
        tgt_lang = row["tgt_lang"]
        src = row["src"]
        translation_match = re.search(f"{src}\n{tgt_lang}:(.*)\n",tokenizer.decode(y))
        if translation_match:
            df.loc[i, "mt"] = translation_match.group(1)
    df.to_csv(output_csv_file, index=False)


if __name__ == "__main__":
    CLI(main_translate)

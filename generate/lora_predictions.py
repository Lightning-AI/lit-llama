import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import tqdm
import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lora import main
from lit_llama import Tokenizer, LLaMA
from lit_llama.lora import lora
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from scripts.prepare_conversations_as_instruction import generate_prompt


def predictions(
    destination_path: Path = Path("/workspace/lit-llama/data/mediform_bookingnegotiation"),
    data_file_name: str = "230531_dialogs_mini.json", # "test.pt", # or "230531_dialogs.json",
    lora_path: Path = Path("/workspace/lit-llama/out/lora/mediform_bookingnegotiation/lit-llama-lora-finetuned.pth"),
    quantize: Optional[str] = None,
    dtype: str = "float32",
    max_new_tokens: int = 100,
) -> None:
    """Predicts for all examples in data_file_name.

    Args:
        destination_path: data directory, e.g. "/workspace/lit-llama/data/mediform_bookingnegotiation"
        data_file_name: json or pt file containing the data points with instruction and output,
            e.g. "test.pt".
        lora_path: the path to the pth file containing the lora adapter weights, e.g.
            "/workspace/lit-llama/out/lora/mediform_bookingnegotiation/lit-llama-lora-finetuned.pth"
        quantize: Whether to quantize the model and using which method, e.g. None or
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        dtype: The dtype to use during generation, e.g. "float32".
        max_new_tokens: The number of generation steps to take, e.g. 100.
    """
    result = []
    destination_path.mkdir(parents=True, exist_ok=True)
    read_path = destination_path / data_file_name
    write_path = destination_path / f'prediction_{data_file_name}'
    if write_path.exists():
        raise FileExistsError(f"The file '{write_path}' already exists.")
    if read_path.suffix == ".json":
        with read_path.open("r") as file:
            data = json.load(file)
    elif read_path.suffix == ".pt":
        data = torch.load(read_path.as_posix())
        data = [{'instruction': sample['instruction'], 'output': sample['output']} for sample in data]
    else:
        exit(f"Files of type {read_path.suffix} cannot be read")
    for index, sample in tqdm.tqdm(enumerate(data)):
        if 'prediction' not in sample:
            generation = main(prompt=sample['instruction'], lora_path=lora_path, quantize=quantize, dtype=dtype, max_new_tokens=max_new_tokens, top_k=1, temperature=0.1)
            sample['prediction'] = get_generated_part(sample['instruction'], generation)
        result.append(sample)
        if index % 10 == 9 or index == len(data) - 1:
            with write_path.open('w') as file:
                json.dump(result, file, indent=2)


def get_generated_part(instruction: str, generation: str) -> str:
    # due to reformatting, slice `len(instruction):`` does not work robustly
    search_window_size = min(10, len(instruction))
    instruction_suffix = instruction[-5:]
    prediction_search_window = generation[len(instruction)-search_window_size:len(instruction)+search_window_size]
    last_index_instruction_suffix_in_window = prediction_search_window.rfind(instruction_suffix)
    if last_index_instruction_suffix_in_window == -1:
        print(f'Warning: String {instruction_suffix} does not occur in search window {prediction_search_window}')
        return generation[len(instruction):].strip()
    if last_index_instruction_suffix_in_window != prediction_search_window.find(instruction_suffix):
        print(f'Warning: String {instruction_suffix} occurs multiple times in search window {prediction_search_window}')
        return generation[len(instruction):].strip()
    return generation[len(instruction)-search_window_size+last_index_instruction_suffix_in_window+len(instruction_suffix):].strip()


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(predictions)

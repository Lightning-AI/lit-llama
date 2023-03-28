# MIT License

# Copyright (c) 2022 Andrej Karpathy

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import sys
from pathlib import Path

import torch
import requests
import json
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm


DATA_FILE = "https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned.json"
DATA_FILE_NAME = "alpaca_data_cleaned.json"
IGNORE_INDEX = -100


def download(file_path: Path):
    # download the (cleaned) alpaca dataset
    if file_path.exists():
        return
    with open(file_path, "w") as f:
        f.write(requests.get(DATA_FILE).text)


def prepare(
    destination_path: Path = Path("data/alpaca"), 
    test_split_size: int = 2000,
    seed: int = 21,
) -> None:
    """Prepare the Alpaca dataset."""
    
    destination_path.mkdir(parents=True, exist_ok=True)
    file_path = destination_path / DATA_FILE_NAME
    download(file_path)

    # TODO: If we don't have the Meta weights, where do we get the tokenizer from? Maybe HF
    tokenizer = Tokenizer("checkpoints/lit-llama/tokenizer.model")
    
    with open(file_path, "r") as file:
        data = json.load(file)

    # Partition the dataset into train and test
    train_split_size = len(data) - test_split_size
    train_set, test_set = random_split(
        data, 
        lengths=(train_split_size, test_split_size),
        generator=torch.Generator().manual_seed(seed),
    )
    train_set, test_set = list(train_set), list(test_set)
        
    print(len(train_set), len(test_set))

    print("Processing train split ...")
    train_set = [generate_and_tokenize_prompt(tokenizer, d) for d in tqdm(train_set)]
    print("Processing test split ...")
    test_set = [generate_and_tokenize_prompt(tokenizer, d) for d in tqdm(test_set)]

    print(train_set[0])
    
    torch.save(train_set, file_path.parent / "train.pt")
    torch.save(test_set, file_path.parent / "test.pt")


def generate_and_tokenize_prompt(tokenizer: Tokenizer, example: dict):
    full_prompt = generate_prompt(example)

    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=256)  # TODO: parameterize this
    encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    return {**example, "input_ids": encoded_full_prompt_and_response, "labels": labels}


def tokenize(tokenizer: Tokenizer, prompt: str, max_length: int) -> torch.Tensor:
    encoded = tokenizer.encode(
        prompt,
        bos=True,
        eos=True,
        truncate=True,
        max_length=max_length,
        # padding=False,
    )
    # if (
    #     encoded[-1].item() != tokenizer.eos_id
    #     and len(encoded) < cutoff_len
    #     and add_eos_token
    # ):
    #     encoded = torch.cat((encoded, torch.tensor([tokenizer.eos_id], dtype=encoded.dtype)))
    return encoded


def generate_prompt(example):
    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


if __name__ == "__main__":
    # support running without installing as a package
    wd = Path(__file__).parent.parent.resolve()
    sys.path.append(str(wd))

    from jsonargparse import CLI

    CLI(prepare)



# alpaca lora example
"""
{'input_ids': tensor([    2, 45943,    16,    41, 15741,    14,  7448,    10,  3685,     4,
        21062,    10,  1263,    14, 16574, 25830,     5,  2069,     4, 50118,
        50118, 48134, 41241,    35, 50118, 31033,   130,  4965,    13,  4959,
         2245,     4, 50118, 50118, 48134, 19121,    35,   134,     4, 21213,
           10,  9320,  5626,     8,   146,   686,     7,   680,  2710,     9,
        12849,     8,  8942,     4,  1437, 50118,   176,     4, 30450,  4595,
            7,   489,   110,   809,  2171,     8,   670,     4,  1437, 50118,
          246,     4,  2315,   615,  3581,     8,  3014,    10,  4292,  3581,
         3078,     4,     2]), 'labels': tensor([ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,   134,     4, 21213,
           10,  9320,  5626,     8,   146,   686,     7,   680,  2710,     9,
        12849,     8,  8942,     4,  1437, 50118,   176,     4, 30450,  4595,
            7,   489,   110,   809,  2171,     8,   670,     4,  1437, 50118,
          246,     4,  2315,   615,  3581,     8,  3014,    10,  4292,  3581,
         3078,     4,     2])}

"""
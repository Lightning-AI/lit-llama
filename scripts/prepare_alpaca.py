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
    
    torch.save(train_set, file_path.parent / "train.pt")
    torch.save(test_set, file_path.parent / "test.pt")


def tokenize(tokenizer, prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    # yeah me too, gotta move fast too


    cutoff_len= 256

    encoded = tokenizer.encode(
        prompt,
        bos=False,  # bos needed?
        eos=False,
        # truncation=True,
        # max_length=cutoff_len= 256,,
        # padding=False,
        # return_tensors=None,
    )
    encoded = encoded[:cutoff_len]
    if (
        encoded[-1].item() != tokenizer.eos_id
        and len(encoded) < cutoff_len
        and add_eos_token
    ):
        encoded = torch.cat((encoded, torch.tensor([tokenizer.eos_id], dtype=encoded.dtype)))

    return encoded


def generate_prompt(data_point):
    if data_point["input"]:
        prompt_input = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{data_point['instruction']}\n\n### Input:\n{data_point['input']}\n\n### Response:"
        )
    else:
        prompt_input = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{data_point['instruction']}\n\n### Response:"
        )
    # TODO: should the output be added here?
    return prompt_input

#     if data_point["input"]:
#         return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
# ### Instruction:
# {data_point["instruction"]}
# ### Input:
# {data_point["input"]}
# ### Response:
# {data_point["output"]}"""
#     else:
#         return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501
# ### Instruction:
# {data_point["instruction"]}
# ### Response:
# {data_point["output"]}"""


def generate_and_tokenize_prompt(tokenizer, data_point):
    full_prompt = generate_prompt(data_point)
        # prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        #     for example in list_data_dict
        # ]
        # targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
    # print(list(data_point.keys()))
    full_prompt += data_point["output"]
    encoded_full_prompt = tokenize(tokenizer, full_prompt)
    return {**data_point, "input_ids": encoded_full_prompt, "labels": encoded_full_prompt}



if __name__ == "__main__":
    # support running without installing as a package
    wd = Path(__file__).parent.parent.resolve()
    sys.path.append(str(wd))

    from jsonargparse import CLI

    CLI(prepare)

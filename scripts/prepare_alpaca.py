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
IGNORE_INDEX = -1


def prepare(
    destination_path: Path = Path("data/alpaca"), 
    test_split_size: int = 2000,
    max_seq_length: int = 256,
    seed: int = 42,
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

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [prepare_sample(sample, tokenizer, max_seq_length) for sample in tqdm(train_set)]
    torch.save(train_set, file_path.parent / "train.pt")

    print("Processing test split ...")
    test_set = [prepare_sample(sample, tokenizer, max_seq_length) for sample in tqdm(test_set)]
    torch.save(test_set, file_path.parent / "test.pt")


def download(file_path: Path):
    if file_path.exists():
        return
    with open(file_path, "w") as f:
        f.write(requests.get(DATA_FILE).text)


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int):
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    return {**example, "input_ids": encoded_full_prompt_and_response, "labels": labels}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=True, max_length=max_length)


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

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

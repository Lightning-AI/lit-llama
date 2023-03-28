import torch
from datasets import load_dataset
from lit_llama.tokenizer import Tokenizer
from torch.utils.data import random_split
import json


tokenizer = Tokenizer("checkpoints/lit-llama/tokenizer.model")
val_set_size = 2000

def tokenize(prompt, add_eos_token=True):
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

    result = dict(
        input_ids=encoded,
        attention_mask=[],  # needed?
        labels=encoded.clone(),
    )

    return result


def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501
### Instruction:
{data_point["instruction"]}
### Response:
{data_point["output"]}"""


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return {**data_point, **tokenized_full_prompt}


def new():
    with open("data/alpaca/alpaca_data_cleaned.json", "r") as file:
        data = json.load(file)
    print(len(data))
    print(data[0].keys())

    data = [generate_and_tokenize_prompt(d) for d in data]
    print(data[0])
    print(list(data[0].keys()))
    # ['input_ids', 'attention_mask', 'labels']

    # torch.save("alpaca_data_cleaned_preprocessed.pt")

    # datasetrandom_split(dataset, lenghts, generator, )


def old():
    dataset = load_dataset("json", data_files="data/alpaca/alpaca_data_cleaned.json")
    print(dataset)
    print(dataset["train"][0])
    # {'instruction': 'Give three tips for staying healthy.', 'input': '', 'output': '1. Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule.'}


    train_val = dataset["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = (
        train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    )


    print(train_data[0])
    print(list(train_data[0].keys()))
    # ['instruction', 'input', 'output', 'input_ids', 'attention_mask', 'labels']


if __name__ == "__main__":
    new()
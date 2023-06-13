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
from lit_llama import Tokenizer, LLaMA
from lit_llama.lora import lora
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from scripts.prepare_conversations_as_instruction import generate_prompt

MAX_ADDITIONAL_ARGS = 2

def evaluation(
    destination_path: Path = Path("/workspace/lit-llama/data/mediform_bookingnegotiation"),
    data_file_name: str = "prediction_230531_dialogs_mini.json", # "test.pt", # or "230531_dialogs.json",
) -> None:
    """Evaluations for all examples in data_file_name, treating a prediction as a
    kind of multiclass multioutput classification of its chat history.
    A prediction is parsed into a set of calls, each having a sequence of arguments

    Args:
        destination_path: data directory, e.g. "/workspace/lit-llama/data/mediform_bookingnegotiation"
        data_file_name: json or pt file containing the data points with instruction and output,
            e.g. "prediction_test.pt".
    """
    evaluation = {"total_samples": 0, "total_calls": 0, "bad_predicted_call": {}}
    destination_path.mkdir(parents=True, exist_ok=True)
    read_path = destination_path / data_file_name
    write_path = destination_path / f'evaluation_{data_file_name}'
    if write_path.exists():
        raise FileExistsError(f"The file '{write_path}' already exists.")
    with read_path.open("r") as file:
        data = json.load(file)
    for index, sample in tqdm.tqdm(enumerate(data)):
        evaluate_sample(sample, evaluation)
        if index % 10 == 9 or index == len(data) - 1:
            with write_path.open('w') as file:
                json.dump(evaluation, file, indent=2)
    print(evaluation)


def evaluate_sample(sample, evaluation) -> None:
    evaluation["total_samples"] += 1
    gt_calls = parse_calls(sample["output"])
    prediction_calls = parse_calls(sample["prediction"])
    for call in gt_calls:
        evaluation["total_calls"] += 1
        if call not in evaluation:
            number_args = len(parse_args(gt_calls, call))
            evaluation[call] =  {
                "call_missed": 0,
                "call_hit": 0,
                "predicted_calls_instead": [],
                "arg_missed": {index: 0 for index in range(number_args)},
                "arg_hit": {index: 0 for index in range(number_args)},
                "arg_wrong": {index: 0 for index in range(number_args)},
                "predicted_arg_instead": {index: [] for index in range(number_args)},
                "arg_exceeded": {index: 0 for index in range(number_args, number_args + MAX_ADDITIONAL_ARGS)},
                "predicted_arg_exceeded": {index: [] for index in range(number_args + MAX_ADDITIONAL_ARGS)},
            }
        if call not in prediction_calls:
            evaluation[call]["call_missed"] += 1
            evaluation[call]["predicted_calls_instead"].append({"prediction": sample["prediction"], "user": get_user(sample), "gt": sample['output']})
        else:
            evaluation[call]["call_hit"] += 1
            gt_arguments = parse_args(gt_calls, call)
            assert len(gt_arguments) <= 2, f"GT contains call with unexpected number of arguments ({gt_arguments}) for {call}({gt_calls[call]})"
            prediction_arguments = parse_args(prediction_calls, call)
            del prediction_calls[call]
            for index, arg in enumerate(gt_arguments):
                if index >= len(prediction_arguments):
                    evaluation[call]["arg_missed"][index] += 1
                elif arg != prediction_arguments[index]:
                    evaluation[call]["arg_wrong"][index] += 1
                    evaluation[call]["predicted_arg_instead"][index].append({"prediction": prediction_arguments[index], "user": get_user(sample), "gt": arg})
                else:
                    evaluation[call]["arg_hit"][index] += 1
            assert len(prediction_arguments) <= len(gt_arguments) + MAX_ADDITIONAL_ARGS, f"hallucinated unexpected number of arguments ({len(prediction_arguments)})"
            for index in range(len(gt_arguments), len(prediction_arguments)):
                evaluation[call]["arg_exceeded"][index] += 1
                evaluation[call]["predicted_arg_exceeded"][index].append({"prediction": prediction_arguments[index], "user": get_user(sample)})
    if prediction_calls:
        for call in prediction_calls:
            evaluation["total_calls"] += 1
            if call not in evaluation["bad_predicted_call"]:
                evaluation["bad_predicted_call"][call] = {"sum": 0, "for": {}}
            evaluation["bad_predicted_call"][call]["sum"] += 1
            evaluation["bad_predicted_call"][call]["for"][sample['output']] = evaluation["bad_predicted_call"][call]["for"].get(sample['output'], 0) + 1


def get_user(sample):
    # return [-2].split('user:')[-1].strip()
    return [assistant_user_pair.split('user:')[-1].strip() for assistant_user_pair in sample['prediction'].split('assistant:')]


def parse_args(call_dict, call):
    if call not in ['pre', 'msg', 'getVacancies', 'transfer', 'cronSpec']:
        return []
    args_string = call_dict[call]
    result = re.findall(r"[^' ,]+|'[^']+'", args_string)
    if result[-1] == '':
        result = result[:-1]
    return result


def parse_calls(call_string: str):
    result = {}
    index_for_non_call_text = 0
    cursor = 0

    matches = re.finditer(r"(\w+)\((.*?)\)", call_string)
    for match in matches:
        start, end = match.span()
        if start != cursor:
            non_call_text = call_string[cursor:start].strip()
            if non_call_text:
                result[str(index_for_non_call_text)] = non_call_text
                index_for_non_call_text += 1
        function_name = match.group(1)
        function_args = match.group(2)
        result[function_name] = function_args
        cursor = end
    if cursor != len(call_string):
        non_call_text = call_string[cursor:].strip()
        if non_call_text:
            result[str(index_for_non_call_text)] = non_call_text
    return result


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(evaluation)

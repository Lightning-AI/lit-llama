import tqdm
import random
import pandas as pd
from jsonargparse import CLI
from datasets import load_dataset
from pandas import DataFrame, Series
from langchain import PromptTemplate, FewShotPromptTemplate

example_prompt = PromptTemplate(
    template="{src_lang}: {src}\n{tgt_lang}: {ref}",
    input_variables = ["src_lang", "src", "tgt_lang", "ref"],
)


def main(
    dataset: str = "wmt19",
    pair: str = "de-en",
    output_file: str = "./data/wmt19_de-en.csv",
    ):
    """
    Args:
        dataset: dataset name
        pair: language pair
        output_file: output file
    """
    # load dataset
    print(f"Loading dataset {dataset} {pair}, note that if it is not cached it may take over an hour.")
    dataset = load_dataset(dataset, pair)
    src_lang, tgt_lang = pair.split("-")
    dev_samples = []
    for sample in tqdm.tqdm(dataset["validation"], desc="Processing samples"):
        # wmt dataset from huggingface format
        sample = sample["translation"]
        # generate fewshot examples randomly sampling the train split
        examples = [
            {
                "src_lang": src_lang,
                "src": dataset["train"][sid]["translation"][src_lang],
                "tgt_lang": tgt_lang,
                "ref": dataset["train"][sid]["translation"][tgt_lang]
            }
            for sid in random.sample(range(len(dataset["train"])), k=5)
        ]
        # build fewshot prompt
        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            # prefix="Give the antonym of every input",
            suffix="{src_lang}: {src}\n{tgt_lang}:",
            input_variables=["src_lang", "src", "tgt_lang"],
            example_separator="\n",
        )
        # save in csv format
        try:
            dev_samples.append({
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "src": sample[src_lang],
                "ref": sample[tgt_lang],
                "input": few_shot_prompt.format(
                    src_lang=src_lang,
                    src=sample[src_lang],
                    tgt_lang=tgt_lang,
                ),
                "mt": ""
            })
        except:
            Warning(f"Skipping sample {sample}")
            continue
    # create df and save to csv
    print(f"Saving to {output_file}")
    df = pd.DataFrame(dev_samples)
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    CLI(main)
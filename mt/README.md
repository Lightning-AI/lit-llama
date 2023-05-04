# Using `llama-lit` on translation tasks

Scripts, utilities and documentation required to use these Llama models for translation.

Reminder: `source .venv/bin/activate`

## Download and create fewshot prompts for WMT dev data `wmt_to_prompt_csv.py`

Downloads the dev split of a wmt dataset [from huggingface](https://huggingface.co/datasets?sort=downloads&search=wmt) into a csv with the following columns:

| src_lang        | tgt_lang        | src             | ref                   | input                | mt           |
| --------------- | --------------- | --------------- | --------------------- | -------------------- | ------------ |
| source language | target language | source sentence | reference translation | prompt (with source) | model output |

The prompt will be 5 randomly selected examples from the train split:

```
src_lang: src-train
tgt_lang: ref-train
... (x5)
src_lang: src
tgt_lang:
```

Example call:
`python wmt_to_prompt_csv.py --dataset wmt19 --pair de-en --output_file /data/wmt19_de-en.csv`

## Generate translations from a csv of prompts `translate.py`

Example call (run from the root directory):

`CUDA_VISIBLE_DEVICES=1 python -m mt.translate --input_csv_file "./mt/data/wmt19_en-de.csv" --output_csv_file "./mt/data/wmt19_en-de_translation.csv" --checkpoint_path "./checkpoints/lit-llama/7B/lit-llama.pth" --tokenizer_path "./checkpoints/lit-llama/tokenizer.model" --temperature 0.5 --top_k 200 --max_new_tokens 128`

!<o>! For temperature 0 there is a bug during generation.

Fills in the `mt` column of the csv with the model output and saves it to the output csv file.

## Evaluates translations from csv `eval_translation.py`

Inputs:
- input_csv_file: csv that `translate.py` outputs
- output_csv_file: where to save the eval at sentence level (same cs with added columns for each metric)
- results_json_file: a summary of the results in a compact format (kept in repo)

`CUDA_VISIBLE_DEVICES=0 python -m mt.eval_translation --input_csv_file "./mt/data/wmt19_en-de_translation.csv" --output_csv_file "./mt/data/wmt19_en-de_translation_eval.csv" --results_json_file "./mt/results/wmt19_en-de_results.json"`

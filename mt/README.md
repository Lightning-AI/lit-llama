# Using `llama-lit` on translation tasks

Scripts, utilities and documentation required to use these Llama models for translation.

### `wmt_to_prompt_csv.py`

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

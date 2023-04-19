## Downloading pretrained weights

Except for when you are training from scratch, you will need the pretrained weights from Meta.

### Original source

Download the model weights following the instructions on the official [LLaMA repository](https://github.com/facebookresearch/llama).

Once downloaded, you should have a folder like this:

```text
checkpoints/llama
├── 7B
│   ├── checklist.chk
│   ├── consolidated.00.pth
│   └── params.json
├── 13B
│   ...
├── tokenizer_checklist.chk
└── tokenizer.model
```

Convert the weights to the Lit-LLaMA format:

```bash
python scripts/convert_checkpoint.py --model_size 7B
```

### Alternative sources

You might find the weights hosted online in the HuggingFace hub. Beware that this infringes the original weight's license.

You could try downloading them by running the following command with a specific repo id:

```bash
python scripts/download.py --repo_id 'REPO_ID' --model_size 7B
```

Convert the weights to the Lit-LLaMA format:

```bash
python scripts/convert_hf_checkpoint.py --model_size 7B
```

You are all set. Now you can continue with inference or finetuning.

Try running [`generate.py` to test the imported weights](inference.md).

# Lit-LLaMA User Guide

With Lit-LLaMA you can do these things:

- [Inference:](#inference) Load a checkpoint and generate predictions from the model
- [Finetuning:](#finetuning) Continue training the model on a downstream task
- [Training from scratch:](#training-from-scratch) Train the model from scratch on your own dataset

## Downloading pretrained weights

Except for when you are training from scratch, you will need the pretrained weights from Meta.
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
python scripts/convert_checkpoint.py \
    --output_dir checkpoints/lit-llama \
    --ckpt_dir checkpoints/llama \
    --tokenizer_path checkpoints/llama/tokenizer.model \
    --model_size 7B
```

You are all set. Now you can continue with inference or finetuning.

## Inference

## Finetuning

### Adapter

[article](https://lightning.ai/pages/community/article/understanding-llama-adapters/)

### LoRA

## Training from scratch


## Resources

- Luca's blog post
- Sebastian's blog post
- LLaMA paper
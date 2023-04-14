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

## Finetuning

We provide a simple training scripts in `finetune_lora.py` and `finetune_adapter.py` that instruction-tunes a pretrained model on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset using the techniques of [LoRA](https://arxiv.org/abs/2106.09685) and [Adapter](https://arxiv.org/abs/2303.16199).





### Adapter



[article](https://lightning.ai/pages/community/article/understanding-llama-adapters/)



1. Download the data and generate a instruction tuning dataset:

   ```bash
   python scripts/prepare_alpaca.py
   ```

2. Run the finetuning script

   ```bash
   python finetune_lora.py
   ```
   or 
   ```bash
   python finetune_adapter.py
   ```

It is expected that you have downloaded the pretrained weights as described above.
The finetuning requires at least one GPU with ~24 GB memory (GTX 3090). Follow the instructions in the script to efficiently fit your GPU memory.
Note: For some GPU models you might need to set `torch.backends.cuda.enable_flash_sdp(False)` (see comments at the top of the script).


## Training from scratch


## Resources

- Luca's blog post
- Sebastian's blog post
- LLaMA paper
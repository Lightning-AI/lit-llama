# Finetuning with LoRA

[Low-rank adaption (LoRA)](https://arxiv.org/abs/2106.09685) is a technique to approximate the update to the linear layers in a LLM with a low-rank matrix factorization. This significantly reduces the number of trainable parameters and speeds up training with little impact on the final performance of the model.
We demonstrate this method by instruction-finetuning LLaMA 7B on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset on a **single GTX 3090 (24GB) GPU**.

## Preparation

The steps here only need to be done once:

1. Follow the instructions in the [README](README.md) to install the dependencies.
2. Download and convert the weights and save them in the `./checkpoints` folder as described [here] (#downloading-pretrained-weights).
3. Download the data and generate a instruction tuning dataset:

   ```bash
   python scripts/prepare_alpaca.py
   ```

## Running the finetuning

```bash
python finetune_lora.py
```

The finetuning requires at least one GPU with ~24 GB memory (GTX 3090).
This script will save checkpoints periodically to the folder `out/`.

## Troubleshooting

If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).

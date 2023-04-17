# Finetuning with Adapter

[LLaMA-Adapter](https://arxiv.org/abs/2303.16199) is a form of prefix-tuning that prepends a learnable adaption-prompt to the inputs of the attention blocks in LLaMA. In total, there are only 1.2M parameters to update during finetuning, which significantly reduces the memory footprint and speeds up training.

We are able to demonstrate instruction-finetuning Lit-LLaMA 7B on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset on a **single GTX 3090 (24GB) GPU**. If using 8 GPUs, finetuning can be completed in under 1 hour.

## Preparation

The steps here only need to be done once:

1. Follow the instructions in the [README](README.md) to install the dependencies.
2. Download and convert the weights and save them in the `./checkpoints` folder as described [here] (#downloading-pretrained-weights).
3. If you want to utilize more than one GPU, you should `pip install deepspeed`.
4. Download the data and generate a instruction tuning dataset:

   ```bash
   python scripts/prepare_alpaca.py
   ```

## Running the finetuning

```bash
python finetune_adapter.py
```

The finetuning requires at least one GPU with ~24 GB memory (GTX 3090).
You can speed up training by setting the `devices` variable in the script to utilize more GPUs if available.
Depending on the available GPU memory, you can also tune the `micro_batch_size` parameter to utilize the GPU efficiently.
This script will save checkpoints periodically to the folder `out/`.

## Test the model

You can test the finetuned model with your own instructions by running:

```bash
python generate_adapter.py \
    --prompt "Recommend a movie to watch on the weekend." \
    --dtype bfloat16 \
    --quantize llm.int8
```
Output:
```
A good movie to watch on the weekend would be The Lion King, since it's a classic family film that everyone can enjoy...
```


## Troubleshooting

If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).

# Finetuning with Adapter v2

[LLaMA-Adapter v2](https://arxiv.org/abs/2304.15010) is a form of prefix-tuning that prepends a learnable adaption-prompt to the inputs of the attention blocks in LLaMA. In total, there are only ~4 M parameters to update during finetuning, which significantly reduces the memory footprint and speeds up training.

We are able to demonstrate instruction-finetuning Lit-LLaMA 7B on the [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset on a **single RTX 3090 (24GB) GPU**. If using 8 GPUs, finetuning can be completed in under 1 hour.

If you are new to LLaMA-Adapter and are interested to learn more about how it works before proceeding with the finetuning guide below, you might find our article [Understanding Parameter-Efficient Finetuning of Large Language Models: From Prefix Tuning to LLaMA-Adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/) helpful.

## LLaMA-Adapter v1 versus LLaMA-Adapter v2

LLaMA-Adapter v2 extends the original LLaMA-Adapter idea by adding trainable bias and scale parameters to each linear layer in the transformer. Furthermore, LLaMA-Adapter v2 makes the normalization layers trainable. Where the 7B LLaMA model has 1.2M trainable parameters with LLaMA v1, LLaMA-Adapter v2 adds 2.8 M trainable parameters for the bias and scale parameters and ~300k trainable parameters for the normalization layers. So, adapter v2 has ~4.3 M trainable parameters in total.

If you are interested in using the more lightweight LLaMA-Adapter v1 approach, see [the related LLaMA Adapter how-to doc here](./finetune_adapter.md).

While LLaMA-Adapter v2 increases the number of trainable parameters from 1.2 M (from LLaMA-Apdapter v1) to 4.3 M, the inference cost is not significantly impacted. This is because the additional bias and scale parameters are cheap to compute in the forward pass, and the RMSNorm parameters are already included in the base model. In LLaMA-Adapter v1, the RMSNorm parameters are not trainable.


## Preparation

The steps here only need to be done once:

1. Follow the instructions in the [README](README.md) to install the dependencies.
2. Download and convert the weights and save them in the `./checkpoints` folder as described [here](download_weights.md).
3. If you want to utilize more than one GPU, you should `pip install deepspeed`.
4. Download the data and generate the Alpaca instruction tuning dataset:

   ```bash
   python scripts/prepare_alpaca.py
   ```

   or [prepare your own dataset](#tune-on-your-dataset).

See also: [Finetuning on an unstructured dataset](unstructured_dataset.md)

## Running the finetuning

```bash
python finetune/adapter_v2.py
```

The finetuning requires at least one GPU with ~24 GB memory (RTX 3090).
You can speed up training by setting the `devices` variable in the script to utilize more GPUs if available.
Depending on the available GPU memory, you can also tune the `micro_batch_size` parameter to utilize the GPU efficiently.

For example, the following settings will let you finetune the model in under 1 hour using DeepSpeed Zero-2:

```python
devices = 8
micro_batch_size = 8
```

This script will save checkpoints periodically to the folder `out/`.

> **Note**
> All scripts support argument [customization](customize_paths.md)

## Test the model

You can test the finetuned model with your own instructions by running:

```bash
python generate/adapter_v2.py \
    --prompt "Recommend a movie to watch on the weekend." \
    --quantize llm.int8
```
Output:
```
A good movie to watch on the weekend would be The Lion King, since it's a classic family film that everyone can enjoy...
```
If your GPU supports `bfloat16`, the script will automatically use it. Together with `--quantize llm.int8`, this brings the memory consumption down to ~8 GB.

## Tune on your dataset

With only a few modifications, you can prepare and train on your own instruction dataset.

1. Create a json file in which each row holds one instruction-response pair. 
   A row has an entry for 'instruction', 'input', and 'output', where 'input' is optional an can be 
   the empty string if the instruction doesn't require a context. Below is an example json file:

    ```
    [
        {
            "instruction": "Arrange the given numbers in ascending order.",
            "input": "2, 4, 0, 8, 3",
            "output": "0, 2, 3, 4, 8"
        },
        ...
    ]
    ```

2. Make a copy of `scripts/prepare_alpaca.py` and name it what you want:

    ```bash
    cp scripts/prepare_alpaca.py scripts/prepare_mydata.py
    ```

3. Modify `scripts/prepare_mydata.py` to read the json data file.
4. Run the script to generate the preprocessed, tokenized train-val split:

    ```bash
    python scripts/prepare_mydata.py --destination_path data/mydata/
    ```

5. Run `finetune/adapter_v2.py` by passing in the location of your data (and optionally other parameters):

    ```bash
    python finetune/adapter_v2.py --data_dir data/mydata/ --out_dir out/myexperiment
    ```


## Troubleshooting

If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).

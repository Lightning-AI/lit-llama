# Inference

We demonstrate how to run inference (next token prediction) with the LLaMA base model in the [`generate.py`](generate.py) script:

```bash
python generate.py --prompt "Hello, my name is"
```
Output:
```
Hello my name is TJ. I have a passion for the outdoors, love hiking and exploring. I also enjoy traveling and learning new things. I especially enjoy long walks, good conversation and a friendly smile.
```

The script assumes you have downloaded and converted the weights and saved them in the `./checkpoints` folder as described [here](download_weights.md).

> **Note**
> All scripts support argument [customization](customize_paths.md)

With the default settings, this will run the 7B model and require ~26 GB of GPU memory (A100 GPU).

## Run Lit-LLaMA on consumer devices

On GPUs with `bfloat16` support, the `generate.py` script will automatically convert the weights and consume about ~14 GB.
For GPUs with less memory, or ones that don't support `bfloat16`, enable quantization (`--quantize llm.int8`):

```bash
python generate.py --quantize llm.int8 --prompt "Hello, my name is"
```
This will consume about ~10 GB of GPU memory or ~8 GB if also using `bfloat16`.
See `python generate.py --help` for more options.

You can also use GPTQ-style int4 quantization, but this needs conversions of the weights first:

```bash
python quantize/gptq.py --output_path checkpoints/lit-llama/7B/llama-gptq.4bit.pth --dtype bfloat16 --quantize gptq.int4
```

GPTQ-style int4 quantization brings GPU usage down to about ~5GB. As only the weights of the Linear layers are quantized, it is useful to also use `--dtype bfloat16` even with the quantization enabled.

With the generated quantized checkpoint generation quantization then works as usual with `--quantize gptq.int4` and the newly generated checkpoint file:

```bash
python generate.py --quantize gptq.int4 --checkpoint_path checkpoints/lit-llama/7B/llama-gptq.4bit.pth
```

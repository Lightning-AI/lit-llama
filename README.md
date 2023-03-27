# ⚡ Lightning LLaMA 🦙

This code base is part of the [Lightning Open LLM Initiative](todolink). We aim to reproduce LLaMA completely in the open together with the community and distribute the code and weights without restrictions.

Lightning LLaMA is

- Easy: The code is minimal, easy to understand and boilerplate-free
- Optimized: It is Lightning-fast and optimized to run on any hardware.
- Community-driven: Everyone can participate in discussions, developing the code or training/fine-tuning
- Open source friendly: The license allows you to take the code and build awesome stuff with it!


## Installation

First, clone the repo:

```bash
git clone https://github.com/Lightning-AI/lightning-llama
cd lightning-llama
```

Create a new Python environment. We recommend using [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html):

```bash
conda create -n llama python=3.10
conda activate llama
```

Install the package:

```bash
pip install -e .
pip install -r requirements.txt
```

You are all set!


## Inference

To generate text predictions, you first need to download the trained model weights following the instructions on the official [LLaMA repository](https://github.com/facebookresearch/llama) from Meta. After you have done that, you should have a folder like this:

```
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

You need to convert these weights to the Lightning LLaMA format by running:

```bash
python scripts/convert_checkpoint.py \
    --output_dir checkpoints/lit-llama \
    --ckpt_dir checkpoints/llama \
    --tokenizer_path checkpoints/llama/tokenizer.model \
    --model_size 7B
```

Now you can run inference:

```bash
python scripts/generate.py \
    --prompt "Hello, my name is"" \
    --checkpoint_path checkpoints/lit-llama/7B \
    --tokenizer_path checkpoints/lit-llama/tokenizer.model
```

This will run using the 7B model and will require roughly 26 GB of GPU memory (A100). If you have a GPU with less memory, you can enable quantization with `--quantize true` which will take longer to load but requires only ~8 GB of memory.

See `python scripts/generate.py --help` for more options.


## Training (coming soon!)

As part of the Lightning open LLM initiative, our goal is to collaboratively train LLaMA from scratch to obtain open-sourced checkpoints that everyone can use without restrictions. The file `train.py` contains a simple training skeleton that will evolve into a fully reproducing training script for LLaMA.

To learn more join the [Open LLM Initiative channel](todo) on our discord server.

## Fine-tuning (coming soon!)

As part of the Lightning open LLM initiative, we will develop scripts for instruction-tuning LLaMA which incorporate the following methods:
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Reinforcement Learning from Human Feedback (RLHF)

Join the [Open LLM Initiative channel](todo) on our discord server to learn more!

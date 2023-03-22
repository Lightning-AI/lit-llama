# adapted from karpathy/minGPT

import torch
from models.llama import Transformer as LLAMA, LLAMA_CONFIG_DICT
from tokenizer.llama import Tokenizer
from lightning import seed_everything


def generate(
    prompt, num_samples=10, steps=20, do_sample=True, top_k=200, temperature=0.8
):
    device = torch.device("cuda")
    checkpoint = torch.load("./data/checkpoints/llama/converted_meta/7B/state_dict.pt")

    with device:
        model = LLAMA(LLAMA_CONFIG_DICT["7B"])
        model.load_state_dict(checkpoint)

    tokenizer = Tokenizer("./data/checkpoints/llama/converted_meta/tokenizer.model")

    model.to(device)
    model.eval()

    # x = torch.zeros(len(prompt), dtype=torch.int, device=device)
    x = tokenizer.encode(prompt, bos=True, eos=False)
    x = torch.tensor(x, dtype=torch.int, device=device)
    x = x.expand(num_samples, -1)

    with torch.no_grad():
        for k in range(num_samples):
            y = model.generate(x, steps, temperature=temperature, top_k=top_k)
            print(tokenizer.decode(y[0].tolist()))


if __name__ == "__main__":
    seed_everything(12334)
    generate(prompt="Hello, my name is", num_samples=10, steps=20)

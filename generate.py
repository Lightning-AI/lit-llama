# adapted from karpathy/minGPT

import torch
from models.llama import Transformer as LLAMA, LLAMA_CONFIG_DICT
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from lightning import seed_everything


def generate(prompt='', num_samples=10, steps=20, do_sample=True, top_k=200, temperature = 0.8):
    device = torch.device('cuda')
    # model = LLAMA.from_pretrained('llama')
    model = LLAMA(LLAMA_CONFIG_DICT["7B"])
    # tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

    model.to(device)
    model.eval()

    if prompt == '':
        # to create unconditional samples...
        # huggingface/transformers tokenizer special cases these strings
        prompt = '<|endoftext|>'
    # encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
    # x = encoded_input['input_ids']
    x = torch.zeros(len(prompt), dtype=torch.int, device=device)

    x = x.expand(num_samples, -1)

    with torch.no_grad():
        for k in range(num_samples):
            y = model.generate(x, steps, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')


if __name__ == '__main__':
    seed_everything(12334)
    generate(prompt='Hello, my name is', num_samples=10, steps=20)

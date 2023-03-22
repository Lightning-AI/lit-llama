# adapted from karpathy/minGPT

import torch
from models.llama import Transformer as LLAMA, LLAMA_CONFIG_DICT
from transformers.models.lama import LlamaTokenizer
from lightning import seed_everything


def generate(prompt='', num_samples=10, steps=20, do_sample=True):
    device = torch.device('cuda')
    # model = LLAMA.from_pretrained('llama')
    model = LLAMA(**LLAMA_CONFIG_DICT["7B"])
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

    model.to(device)
    model.eval()

    if prompt == '':
        # to create unconditional samples...
        # huggingface/transformers tokenizer special cases these strings
        prompt = '<|endoftext|>'
    encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
    x = encoded_input['input_ids']

    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)

    for i in range(num_samples):
        out = tokenizer.decode(y[i].cpu().squeeze())
        print('- ' *80)
        print(out)


if __name__ == '__main__':
    seed_everything(12334)
    generate(prompt='Hello, my name is', num_samples=10, steps=20)

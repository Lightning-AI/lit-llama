import sys
import time
from pathlib import Path
from typing import Optional

import lightning as L
import torch

from lit_llama import LLaMA, Tokenizer, as_8_bit_quantized
from generate import generate
from finetune import generate_prompt
import loralib as lora


def prepare_data(data):
    input_ids = torch.tensor(data["input_ids"], dtype=torch.int64)
    labels = torch.tensor(data["labels"], dtype=torch.int64)

    # max_len = max(len(s) for s in input_ids)
    max_len = len(input_ids) + 4

    def pad_left(x, pad_id):
        # TODO: optimize this to pad to the next multiple of 8 or so?
        n = max_len - len(x)
        return torch.cat((torch.full((n,), pad_id, dtype=x.dtype), x))

    x = pad_left(input_ids, pad_id=0)
    y = pad_left(labels, pad_id=-1)
    return x, y


def main():
   
    checkpoint_path = "out/lora-native-new-hparams/iter-001919-ckpt.pt"
    tokenizer_path = Path("./checkpoints/lit-llama/tokenizer.model")
    

    fabric = L.Fabric(accelerator="cuda", devices=1)
    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    model = LLaMA.from_name("7B")
    model.load_state_dict(torch.load(f'./checkpoints/lit-llama/7B/state_dict.pth'), strict=False)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)
    model.eval()
    model = fabric.setup_module(model)
    tokenizer = Tokenizer(tokenizer_path)
    
    
    # sample = {"instruction": prompt, "input": ""}
    train_data = torch.load("data/alpaca/train_orig.pt")
    
    # prompt = generate_prompt(sample)
    # encoded_prompt = tokenizer.encode(prompt, bos=True, eos=True, device=fabric.device)
    # encoded_prompt = encoded_prompt[None, :]  # add batch dimension
    x, y = prepare_data(train_data[0])
    x, y = fabric.to_device((x, y))
    # print(sample)
    print(tokenizer.decode(x))
    # print(tokenizer.decode(y[5:]))
    

    L.seed_everything(1234)
    # t0 = time.perf_counter()

    # for _ in range(num_samples):
    #     y = generate(
    #         model,
    #         encoded_prompt,
    #         max_new_tokens,
    #         model.config.block_size,  # type: ignore[union-attr,arg-type]
    #         temperature=temperature,
    #         top_k=top_k,
    #     )[0]  # unpack batch dimension
    #     print(tokenizer.decode(y))

    logits = model(x[None, :])
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
    print(loss)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = y[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1), ignore_index=-1)
    print(loss)


    # Shift so that tokens < n predict n
    # shift_logits = logits[..., :-1, :].contiguous()
    # shift_labels = labels[..., 1:].contiguous()
    # # Flatten the tokens
    # loss_fct = CrossEntropyLoss()
    # shift_logits = shift_logits.view(-1, self.config.vocab_size)
    # shift_labels = shift_labels.view(-1)
    # # Enable model parallelism
    # shift_labels = shift_labels.to(shift_logits.device)
    # loss = loss_fct(shift_logits, shift_labels)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)

import os
import time

import lightning as L
import numpy as np
import torch

from lit_llama.adapter import LLaMA, LLaMAConfig


micro_batch_size = 2
block_size = 512

def main():
    fabric = L.Fabric(accelerator="cuda", devices=1)
    fabric.launch()

    train_data, val_data = load_datasets()

    config = LLaMAConfig(block_size=block_size, n_layer=4, n_embd=64, n_head=8)

    with fabric.device:
        model = LLaMA(config)
    
    # torch._dynamo.config.verbose=True
    model = torch.compile(model, dynamic=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=9e-3)
    model, optimizer = fabric.setup(model, optimizer)
    
    for iter_num in range(1000):
        print("begin iter", iter_num)
       

        t0 = time.time()
        input_ids, targets = get_batch(fabric, train_data)
        # targets = torch.randint(0, config.vocab_size, size=(2, 128), dtype=torch.int64, device=fabric.device)
        # print(input_ids.shape, targets.shape)
        logits = model(targets)
        loss = loss_fn(logits, targets)
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        dt = time.time() - t0
        print(f"iter {iter_num}, time: {dt*1000:.2f}ms")


def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss
    

def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        # n = max_len - len(x)
        n = block_size - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets(data_dir: str = "data/alpaca"):
    train_data = torch.load(os.path.join(data_dir, "train.pt"))
    val_data = torch.load(os.path.join(data_dir, "test.pt"))
    return train_data, val_data


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    main()

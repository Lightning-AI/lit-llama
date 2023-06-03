# This adapts SparseGPT process: https://github.com/IST-DASLab/sparsegpt
# E. Frantar et al SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot, https://arxiv.org/abs/2301.00774
# portions copyright by the authors licensed under the Apache License 2.0


import gc
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset


wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama import LLaMA, Tokenizer
from lit_llama.sparsification import SparseGPT

from lit_llama.utils import EmptyInitOnDevice, llama_model_lookup


def get_sample_data():
    traindata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    # heuristic for the data size?
    txt = "\n".join(
        traindata[i]["text"] for i in torch.randperm(len(traindata))[:1000].tolist()
    )
    return txt

@torch.no_grad()
def llama_blockwise_sparsification(
    model, 
    sample_inputs, 
    working_device,
    *,
    sparsity=0,
    prunen=0,
    prunem=0,
    
):
    
    print('Getting Inputs for the first block')
    model.transformer.wte.to(working_device)
    sample_inputs = sample_inputs.to(working_device)
    inps = model.transformer.wte(sample_inputs)
    model.transformer.wte.to("cpu")
    torch.cuda.empty_cache()

    rope_cache = model.build_rope_cache(sample_inputs)
    mask_cache = model.build_mask_cache(sample_inputs)

    print('Starting to sparsify block')
    outs = torch.zeros_like(inps)


    submodules_to_process = [
        "attn.c_attn",
        "attn.c_proj",
        "mlp.c_fc1",
        "mlp.c_fc2",
        "mlp.c_proj",
    ]


    for i, block in enumerate(model.transformer.h):

        block.to(working_device)

        for name in submodules_to_process:
            print(i, name, end=" ")
            t0 = time.perf_counter()
            print("collecting stats", end=" ")
            sys.stdout.flush()
            module = block.get_submodule(name)

            sparsegpt = SparseGPT(
                module,
                sparsity=sparsity,
                prunen=prunen,
                prunem=prunem,
            )
            
            handle = model.lm_head.register_forward_hook(sparsegpt.collect_input_stats)
           
            for j in range(inps.size(0)):
                outs[j : j + 1], _ = block(
                    inps[j : j + 1],
                    rope=rope_cache,
                    mask=mask_cache,
                    max_seq_length=model.config.block_size
                )

            handle.remove()

            error = sparsegpt.sparsify()

            del sparsegpt
            gc.collect()
            torch.cuda.empty_cache()
            t1 = time.perf_counter()
            print(f"time {int(t1 - t0 + 0.5)}s sparsification error {error:.1f}")

        
        for j in range(inps.size(0)):
            outs[j : j + 1], _ = block(
                inps[j : j + 1],
                rope=rope_cache,
                mask=mask_cache,
                max_seq_length=model.config.block_size
            )

        block.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.transformer.ln_f.to(working_device)
    for j in range(inps.size(0)):
        outs[j : j + 1] = model.transformer.ln_f(inps[j : j + 1])
    model.transformer.ln_f.to("cpu")

    # normalised out will be input to the LM head
    inps, outs = outs, inps

    model.lm_head.to(working_device)
    sparsegpt = SparseGPT(
        model.lm_head,
        sparsity=sparsity,
        prunen=prunen,
        prunem=prunem,
    )

    # During the forward pass, the collect_input_stats function collects input statistics and updates the Hessian matrix.
    handle = model.lm_head.register_forward_hook(sparsegpt.collect_input_stats)
    for j in range(inps.size(0)):
        model.lm_head(inps[j : j + 1])
    handle.remove()
    # After the forward pass, the sparsify function can be called to perform the  sparsification based on the collected statistics.
    error = sparsegpt.sparsify()
    model.lm_head.to("cpu")

def main(
    *,
    checkpoint_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth"),
    output_path: Optional[Path] = None,
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    n_samples: int = 128,
    dtype: str = "float32",
    sparsity: int = 0,
    prunem: int = 0,
    prunen: int = 0
) -> None:
    """
       Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        checkpoint_path: The checkpoint path to load.
        output_path: Path to write the sparsified model's state dict to.
        tokenizer_path: The tokenizer path to load.
        n_samples: Number of example inputs to use for statistics (default: 128)
        dtype: The dtype to use to load the model.
        sparsity: Target sparsity
        prunem: M for N:M pruning.
        prunen: N for N:M pruning.
    """
    assert checkpoint_path.is_file()
    assert tokenizer_path.is_file()
    if output_path is None:
        output_path = checkpoint_path.parent / "llama-gpt-sparsified.pth"
    assert output_path.parent.is_dir() and (not output_path.exists() or output_path.is_file())

    device = "cuda"

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt


    # we avoid loading the entire model on the GPU and do this block by block
    with EmptyInitOnDevice(
        device="cpu",
        dtype=dtype,
    ):
        print("Loading model ...", file=sys.stderr)
        t0 = time.time()
        checkpoint = torch.load(checkpoint_path)
        name = llama_model_lookup(checkpoint)
        model = LLaMA.from_name(name)
        model.load_state_dict(checkpoint)
        print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()

    tokenizer = Tokenizer(tokenizer_path)

    test_string = get_sample_data()
    encoded_text = tokenizer.encode(
        test_string,
        bos=True,
        eos=False, 
    )

    block_size = 2048  
    # truncate the text and reshape to batch by sequence length
    encoded_text = encoded_text[: n_samples * block_size].reshape(n_samples, block_size)

    t0 = time.perf_counter()
    llama_blockwise_sparsification(
                                   model=model, 
                                   sample_inputs=encoded_text, 
                                   working_device=device,
                                   sparsity=sparsity,
                                   prunen=prunen,
                                   prunem=prunem
                                   )
    t = time.perf_counter() - t0

    print(
        f"\n\nTime for sparsification: {t:.02f} sec total",
        file=sys.stderr,
    )
    print(
        f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB",
        file=sys.stderr,
    )

    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)


   









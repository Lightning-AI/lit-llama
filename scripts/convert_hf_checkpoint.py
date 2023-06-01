import collections
import contextlib
import gc
import json
import shutil
import sys
from pathlib import Path

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_llama.model import LLaMA, LLaMAConfig
from lit_llama.utils import EmptyInitOnDevice, lazy_load, incremental_save


@torch.no_grad()
def convert_hf_checkpoint(
    *,
    output_dir: Path = Path("checkpoints/lit-llama/7B"),
    checkpoint_dir: Path = Path("checkpoints/hf-llama/7B"),
    model_size: str = "7B",
    dtype: str = "float32",
    verify: bool = False,
) -> None:
    """
    Perform the reverse operation of: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # the tokenizer is the same for all model sizes, so we store it in the parent dir
    shutil.copy(checkpoint_dir / "tokenizer.model", output_dir.parent)

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    print("Initializing lit-llama")
    config = LLaMAConfig.from_name(model_size)

    with EmptyInitOnDevice(device="meta", dtype=dtype):
        model = LLaMA(config)

    qkv_size = model.transformer.h[0].attn.c_attn.weight.shape[0] // 3

    # initialize a new empty state dict to hold our new weights
    sd_meta = model.state_dict()
    sd = {}

    # Load the json file containing weight mapping
    pytorch_bin_map_json_path = checkpoint_dir / "pytorch_model.bin.index.json"
    with open(pytorch_bin_map_json_path) as json_map:
        bin_index = json.load(json_map)
    bin_files = set(checkpoint_dir / bin for bin in bin_index["weight_map"].values())
    if not bin_files:
        raise ValueError(f"Expected {str(checkpoint_dir)!r} to contain .bin files")

    def permute(w):
        dim = config.n_embd
        w = w._load_tensor().to(dtype)
        return (
            w.view(config.n_head, 2, dim // config.n_head // 2, dim)
            .transpose(1, 2)
            .reshape(dim, dim)
        )

    weight_map = {
        "self_attn.o_proj.weight": "attn.c_proj.weight",
        "self_attn.q_proj.weight": "attn.c_attn.weight",
        "self_attn.k_proj.weight": "attn.c_attn.weight",
        "self_attn.v_proj.weight": "attn.c_attn.weight",
        "mlp.gate_proj.weight": "mlp.c_fc1.weight",
        "mlp.up_proj.weight": "mlp.c_fc2.weight",
        "mlp.down_proj.weight": "mlp.c_proj.weight",
        "input_layernorm.weight": "rms_1.scale",
        "post_attention_layernorm.weight": "rms_2.scale",
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.norm.weight": "transformer.ln_f.scale",
        "lm_head.weight": "lm_head.weight",
    }

    print(f"Saving to disk at {output_dir}")
    unprocessed_weights = collections.defaultdict(dict)

    with incremental_save(output_dir / "lit-llama.pth") as saver:
        # for checkpoints that split the QKV across several files, we need to keep all the bin files
        # open, so we use `ExitStack` to close them all together at the end
        with contextlib.ExitStack() as stack:
            for bin_file in bin_files:
                print("Processing", bin_file)
                hf_weights = stack.enter_context(lazy_load(bin_file))
                for name, param in hf_weights.items():
                    skip = False
                    if "rotary_emb.inv_freq" in name:
                        continue
                    if "model.layers" in name:
                        block_id = int(name.split(".")[2])
                        from_name = ".".join(name.split(".")[3:])
                        to_name = weight_map[from_name]
                        sd_key = f"transformer.h.{block_id}.{to_name}"

                        if "q_proj" in name:
                            unprocessed_weights[sd_key]["q_proj"] = param
                            skip = True
                        elif "k_proj" in name:
                            unprocessed_weights[sd_key]["k_proj"] = param
                            skip = True
                        elif "v_proj" in name:
                            unprocessed_weights[sd_key]["v_proj"] = param
                            skip = True
                        if skip and len(unprocessed_weights[sd_key]) == 3:
                            w = torch.empty(
                                sd_meta[sd_key].shape, dtype=sd_meta[sd_key].dtype
                            )
                            w[:qkv_size] = permute(unprocessed_weights[sd_key]["q_proj"])
                            w[qkv_size:-qkv_size] = permute(
                                unprocessed_weights[sd_key]["k_proj"]
                            )
                            w[-qkv_size:] = (
                                unprocessed_weights[sd_key]["v_proj"]
                                ._load_tensor()
                                .to(dtype)
                            )
                            sd[sd_key] = w
                            del unprocessed_weights[sd_key]
                            skip = False
                        else:
                            sd[sd_key] = param._load_tensor().to(dtype)
                    else:
                        sd_key = weight_map[name]
                        sd[sd_key] = param._load_tensor().to(dtype)
                    if not skip:
                        sd[sd_key] = saver.store_early(sd[sd_key])
                gc.collect()
        saver.save(sd)

    assert len(unprocessed_weights) == 0, f"unexpected partial weights {list(unprocessed_weights)}"
    if verify:
        try:
            from transformers import LlamaForCausalLM
        except ImportError:
            raise ImportError("verify=True requires transformers to be installed, please `pip install transformers`")
        print("Verifying...")

        token_sample = torch.randint(0, config.vocab_size, size=(1, config.block_size), dtype=torch.int64)
        out = model(token_sample)
        del model
        gc.collect()

        print("Loading original model for comparison")
        model_hf = LlamaForCausalLM.from_pretrained(checkpoint_dir)
        out_hf = model_hf(token_sample)["logits"]

        print("Comparing outputs")
        assert out.device.type == out_hf.device.type
        assert out.dtype == out_hf.dtype
        assert torch.testing.assert_close(out, out_hf)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_hf_checkpoint)


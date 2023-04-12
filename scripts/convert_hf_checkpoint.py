from pathlib import Path
import sys
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
from lit_llama.model import LLaMA, LLaMAConfig
import os
import json
from transformers import LlamaForCausalLM

def load_weights_from_bin(weight_path: Path, weight_file: str) -> torch.Tensor:
    weights = torch.load(weight_path / weight_file, map_location="cpu")
    return weights

def convert_hf_checkpoint(
    model_size: str = "7B",
    hf_checkpoint_path: Path = Path("checkpoints/llama-7b-hf"),
    lit_checkpoint: Path = Path("checkpoints/lit-llama.ckpt"),
    verify: bool = False,
) -> None:


    """
    Perform the reverse operation of: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
    """

    print("Loading weights from pretrained LLaMA %s" % model_size)

    config = LLaMAConfig.from_name(model_size)

    print("Loaded config %s, please wait for model loading" % config)

    model = LLaMA(config)
    print("Initialized lit-style instance of llama")
    sd = model.state_dict()

    qkv_size = model.transformer.h[0].attn.c_attn.weight.shape[0] // 3
    n_blocks = len(model.transformer.h)

    print("Finished loading model")

    # initialize a new empty state dict to hold our new weights
    sd = model.state_dict()

    # Load the json file containing weight mapping
    pytorch_bin_map_json_path = os.path.join(hf_checkpoint_path, "pytorch_model.bin.index.json")
    with open(pytorch_bin_map_json_path) as json_map:
        bin_index = json.load(json_map)

    def permute(w):
        dim = config.n_embd
        return (
            w.view(config.n_head, 2, dim // config.n_head // 2, dim)
            .transpose(1, 2)
            .reshape(dim, dim)
        )

    with torch.no_grad():

        print("Total blocks to process: %s" % n_blocks)

        for i in range(n_blocks):
            print("Processing block %s" % i)

            for layer_name, bin_file in bin_index["weight_map"].items():
                if f"model.layers.{i}." in layer_name:
                    # Load the weight tensor from the .bin file,

                    hf_weights = load_weights_from_bin(hf_checkpoint_path, bin_file)
                    print(f"Loading layer {i} weights from bin file {bin_file}")
                    sd[f"transformer.h.{i}.attn.c_proj.weight"].copy_(
                        hf_weights[f"model.layers.{i}.self_attn.o_proj.weight"]
                    )

                    sd[f"transformer.h.{i}.attn.c_attn.weight"][:qkv_size] = permute(
                        hf_weights[f"model.layers.{i}.self_attn.q_proj.weight"]
                    )
                    sd[f"transformer.h.{i}.attn.c_attn.weight"][qkv_size:-qkv_size] = permute(
                        hf_weights[f"model.layers.{i}.self_attn.k_proj.weight"]
                    )
                    sd[f"transformer.h.{i}.attn.c_attn.weight"][-qkv_size:] = hf_weights[
                        f"model.layers.{i}.self_attn.v_proj.weight"
                    ]

                    sd[f"transformer.h.{i}.mlp.c_fc1.weight"].copy_(
                        hf_weights[f"model.layers.{i}.mlp.gate_proj.weight"]
                    )
                    sd[f"transformer.h.{i}.mlp.c_fc2.weight"].copy_(
                        hf_weights[f"model.layers.{i}.mlp.up_proj.weight"]
                    )
                    sd[f"transformer.h.{i}.mlp.c_proj.weight"].copy_(
                        hf_weights[f"model.layers.{i}.mlp.down_proj.weight"]
                    )

                    sd[f"transformer.h.{i}.rms_1.scale"].copy_(
                        hf_weights[f"model.layers.{i}.input_layernorm.weight"]
                    )
                    sd[f"transformer.h.{i}.rms_2.scale"].copy_(
                        hf_weights[f"model.layers.{i}.post_attention_layernorm.weight"]
                    )

                    # Load globals, could be done in a cleaner way. It assumes that the globals will happen to be in a .bin
                    # ... file that also contains some layers, hopefully always the case
                    if "model.embed_tokens.weight" in hf_weights:
                        sd["transformer.wte.weight"].copy_(hf_weights["model.embed_tokens.weight"])

                    if "model.norm.weight" in hf_weights:
                        sd["transformer.ln_f.scale"].copy_(hf_weights["model.norm.weight"])

                    if "lm_head.weight" in hf_weights:
                        sd["lm_head.weight"].copy_(hf_weights["lm_head.weight"])

                    # Break here to avoid reloading the same weights multiple times
                    # this assumes the layers won't be split between bin files, which hopefully is not possible
                    break


    if verify:
        print("Verifying...")
        print("Loading huggingface model for comparison. This will use a lot of ram and take a while. You'll certainly run out with less than 64gb of cpu ram")
        model_hf = LlamaForCausalLM.from_pretrained(hf_checkpoint_path)

        token_sample = torch.randint(
            0, config.vocab_size, size=(1, config.block_size), dtype=torch.int64
        )

        with torch.no_grad():
            out = model(token_sample)
            out_hf = model_hf(token_sample)
        print("Comparing tokenization outputs")
        assert torch.allclose(out, out_hf["logits"])

    print("Saving to disk at %s " % lit_checkpoint)
    torch.save(model.state_dict(), lit_checkpoint)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_hf_checkpoint)

# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import shutil

import torch

"""
Sample usage:

```bash
python -m models.llama.convert_checkpoint -h
```
"""

INTERMEDIATE_SIZE_MAP = {
    "7B": 11008,
    "13B": 13824,
    "30B": 17920,
    "65B": 22016,
}
NUM_SHARDS = {
    "7B": 1,
    "13B": 2,
    "30B": 4,
    "65B": 8,
}
META_KEY_TO_DIM = {
    "w1": 0,
    "w2": -1,
    "w3": 0,
    "wo": -1,
    "wq": 0,
    "wk": 0,
    "wv": 0,
    "output": 0,
    "tok_embeddings": -1,
    "ffn_norm": None,
    "attention_norm": None,
    "norm": None,
    "rope": None,
}


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(model_path, input_base_path, model_size):
    assert model_size in INTERMEDIATE_SIZE_MAP
    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(input_base_path, "params.json")) as f:
        params = json.load(f)
    print("Model size:", model_size, "Params:", params)
    num_shards = NUM_SHARDS[model_size]
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = 10000.0
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head)
    )

    # permute for sliced rotary
    def permute(w):
        return (
            w.view(n_heads, dim // n_heads // 2, 2, dim)
            .transpose(1, 2)
            .reshape(dim, dim)
        )

    # Load weights
    if model_size == "7B":
        # Not shared
        # (The sharded implementation would also work, but this is simpler.)
        loaded = torch.load(
            os.path.join(input_base_path, "consolidated.00.pth"), map_location="cpu"
        )
    else:
        # Sharded
        loaded = [
            torch.load(
                os.path.join(input_base_path, f"consolidated.{i:02d}.pth"),
                map_location="cpu",
            )
            for i in range(num_shards)
        ]
    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = "pytorch_model-{:05d}-of-{:05d}.bin".format(
            layer_i + 1,
            n_layers + 1,
        )
        if model_size == "7B":
            # Unsharded
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                    loaded[f"layers.{layer_i}.attention.wq.weight"]
                ),
                f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                    loaded[f"layers.{layer_i}.attention.wk.weight"]
                ),
                f"model.layers.{layer_i}.self_attn.v_proj.weight": loaded[
                    f"layers.{layer_i}.attention.wv.weight"
                ],
                f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[
                    f"layers.{layer_i}.attention.wo.weight"
                ],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[
                    f"layers.{layer_i}.feed_forward.w1.weight"
                ],
                f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[
                    f"layers.{layer_i}.feed_forward.w2.weight"
                ],
                f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[
                    f"layers.{layer_i}.feed_forward.w3.weight"
                ],
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[
                    f"layers.{layer_i}.attention_norm.weight"
                ],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[
                    f"layers.{layer_i}.ffn_norm.weight"
                ],
            }
        else:
            # Sharded
            state_dict = {
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[0][
                    f"layers.{layer_i}.attention_norm.weight"
                ],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[0][
                    f"layers.{layer_i}.ffn_norm.weight"
                ],
            }
            state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wq.weight"].view(
                            n_heads_per_shard, dims_per_head, dim
                        )
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(dim, dim)
            )
            state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wk.weight"].view(
                            n_heads_per_shard, dims_per_head, dim
                        )
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(dim, dim)
            )
            state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wv.weight"].view(
                        n_heads_per_shard, dims_per_head, dim
                    )
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(dim, dim)

            state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wo.weight"]
                    for i in range(num_shards)
                ],
                dim=1,
            )
            state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"]
                    for i in range(num_shards)
                ],
                dim=0,
            )
            state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"]
                    for i in range(num_shards)
                ],
                dim=1,
            )
            state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"]
                    for i in range(num_shards)
                ],
                dim=0,
            )

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(model_path, filename))

    filename = "pytorch_model-{:05d}-of-{:05d}.bin".format(
        n_layers + 1,
        n_layers + 1,
    )
    if model_size == "7B":
        # Unsharded
        state_dict = {
            "model.embed_tokens.weight": loaded["tok_embeddings.weight"],
            "model.norm.weight": loaded["norm.weight"],
            "lm_head.weight": loaded["output.weight"],
        }
    else:
        state_dict = {
            "model.norm.weight": loaded[0]["norm.weight"],
            "model.embed_tokens.weight": torch.cat(
                [loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1
            ),
            "lm_head.weight": torch.cat(
                [loaded[i]["output.weight"] for i in range(num_shards)], dim=0
            ),
        }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(model_path, "pytorch_model.bin.index.json"))
    config_out = {
        "architectures": ["LLaMAForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": 1,
        "hidden_act": "silu",
        "hidden_size": params["dim"],
        "intermediate_size": INTERMEDIATE_SIZE_MAP[model_size],
        "initializer_range": 0.02,
        "max_sequence_length": 2048,
        "model_type": "llama",
        "num_attention_heads": params["n_heads"],
        "num_hidden_layers": params["n_layers"],
        "pad_token_id": -1,
        "rms_norm_eps": params["norm_eps"],
        "torch_dtype": "float16",
        "transformers_version": "4.27.0.dev0",
        "use_cache": True,
        "vocab_size": 32000,
    }
    write_json(
        config_out,
        os.path.join(model_path, "config.json"),
    )
    generation_config = {
        "_from_model_config": True,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 0,
        "transformers_version": "4.27.0.dev0",
    }
    write_json(
        generation_config,
        os.path.join(model_path, "generation_config.json"),
    )


def write_tokenizer(tokenizer_path, input_tokenizer_path):
    os.makedirs(tokenizer_path, exist_ok=True)
    write_json({}, os.path.join(tokenizer_path, "special_tokens_map.json"))
    write_json(
        {
            "bos_token": "",
            "eos_token": "",
            "model_max_length": int(1e30),
            "tokenizer_class": "LLaMATokenizer",
            "unk_token": "",
        },
        os.path.join(tokenizer_path, "tokenizer_config.json"),
    )
    shutil.copyfile(input_tokenizer_path, os.path.join(tokenizer_path, "tokenizer.model"))


def convert_llama_hf(
    *,
    output_dir: str,
    ckpt_dir: str = "/srv/data/checkpoints/llama/raw",
    tokenizer_path: str = "/srv/data/checkpoints/llama/raw/tokenizer.model",
    model_size: str = "7B",
):
    write_model(
        model_path=os.path.join(output_dir, "llama-{}".format(model_size).lower()),
        input_base_path=os.path.join(ckpt_dir, model_size),
        model_size=model_size,
    )
    write_tokenizer(tokenizer_path=os.path.join(output_dir, "tokenizer"), input_tokenizer_path=tokenizer_path)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(convert_llama_hf)

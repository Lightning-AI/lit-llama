from lit_llama.model import LLaMA, LLaMAConfig

from transformers import LlamaForCausalLM
import torch


def from_pretrained(name, checkpoint_path):
    print("Loading weights from pretrained LLaMA %s" % name)

    config = LLaMAConfig.from_name(name)
    model = LLaMA(config)
    sd = model.state_dict()

    model_hf = LlamaForCausalLM.from_pretrained(checkpoint_path)
    sd_hf = model_hf.state_dict()

    qkv_size = model.transformer.h[0].attn.c_attn.weight.shape[0] // 3
    n_blocks = len(model.transformer.h)

    with torch.no_grad():
        sd["transformer.wte.weight"].copy_(sd_hf["model.embed_tokens.weight"])
        sd["transformer.ln_f.scale"].copy_(sd_hf["model.norm.weight"])
        sd["lm_head.weight"].copy_(sd_hf["lm_head.weight"])

        for i in range(n_blocks):
            sd[f"transformer.h.{i}.attn.c_proj.weight"].copy_(sd_hf[f"model.layers.{i}.self_attn.o_proj.weight"])

            sd[f"transformer.h.{i}.attn.c_attn.weight"][:qkv_size] = sd_hf[f"model.layers.{i}.self_attn.q_proj.weight"]
            sd[f"transformer.h.{i}.attn.c_attn.weight"][qkv_size:2*qkv_size] = sd_hf[f"model.layers.{i}.self_attn.k_proj.weight"]
            sd[f"transformer.h.{i}.attn.c_attn.weight"][-qkv_size:] = sd_hf[f"model.layers.{i}.self_attn.v_proj.weight"]

            sd[f"transformer.h.{i}.mlp.c_fc1.weight"].copy_(sd_hf[f"model.layers.{i}.mlp.gate_proj.weight"])
            sd[f"transformer.h.{i}.mlp.c_fc2.weight"].copy_(sd_hf[f"model.layers.{i}.mlp.up_proj.weight"])
            sd[f"transformer.h.{i}.mlp.c_proj.weight"].copy_(sd_hf[f"model.layers.{i}.mlp.down_proj.weight"])

            sd[f"transformer.h.{i}.rms_1.scale"].copy_(sd_hf[f"model.layers.{i}.input_layernorm.weight"])
            sd[f"transformer.h.{i}.rms_2.scale"].copy_(sd_hf[f"model.layers.{i}.post_attention_layernorm.weight"])

    token_sample = torch.randint(
        0, config.vocab_size, size=(1, config.block_size), dtype=torch.int64
    )

    with torch.no_grad():
        out = model(token_sample)
        out_hf = model_hf(token_sample)

    print(out)
    print(out_hf["logits"])
    print((out - out_hf["logits"]).sum())
    assert torch.allclose(out, out_hf["logits"])

    return model


if __name__ == "__main__":
    model = from_pretrained("7B", "../hf-checkpoint/llama-7b-hf")

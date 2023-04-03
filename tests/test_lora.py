from lit_llama.lora import lora, CausalSelfAttention
from lit_llama.model import LLaMA, LLaMAConfig


def test_lora_layer_replacement():
    config = LLaMAConfig()
    config.n_layer = 2
    config.n_head = 4
    config.n_embd = 8
    config.block_size = 8
    config.vocab_size = 8

    with lora(r=8, alpha=8, dropout=0.1):
        model = LLaMA(config)

    # CausalSelfAttention got replaced with its LoRA version
    assert isinstance(model.transformer.h[0].attn, CausalSelfAttention)
    assert isinstance(model.transformer.h[1].attn, CausalSelfAttention)

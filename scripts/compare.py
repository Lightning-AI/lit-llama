import os
import sys
from contextlib import contextmanager

import torch


def compare_rope():
    # with t > 1, assertion fails!
    b, t = 1, 6
    n_embed = 8
    n_head = 2

    x = torch.randint(0, 10000, size=(b, t, n_head, n_embed // n_head)).float()

    freqs_cis = orig_llama.precompute_freqs_cis(n_embed // n_head, t)
    llama_rope_cache = llama.build_rope_cache(t, n_embed // n_head, dtype=x.dtype, device=x.device, base=10000)
    llama_rope_cache2 = llama.build_rope_cache(t, n_embed // n_head, dtype=x.dtype, device=x.device, base=10000)
    assert torch.equal(llama_rope_cache, llama_rope_cache2)

    # llama_x_rope = llama.apply_rope(x, llama_rope_cache)
    from model import rotate_neg_half
    neg_half_x = rotate_neg_half(x)
    cos, sin = llama_rope_cache
    T = x.size(2)
    cos = cos[:, :, :T]
    sin = sin[:, :, :T]
    llama_x_rope = (x * cos) + (neg_half_x * sin)


    # orig_llama_x_rope, _ = orig_llama.apply_rotary_emb(x, x, freqs_cis)
    xq_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    freqs_cis = freqs_cis.view(1, xq_.size(1), 1, xq_.size(3))
    orig_llama_x_rope = torch.view_as_real(xq_ * freqs_cis).flatten(3)

    assert torch.allclose(llama_x_rope, orig_llama_x_rope)



def compare_rmsnorm():
    block_size = 16
    vocab_size = 16

    sample = torch.rand(size=(2, block_size, vocab_size), dtype=torch.float32)

    eps = 1e-6
    orig_llama_rmsnorm = orig_llama.RMSNorm(vocab_size, eps=eps)(sample)
    llama_rmsnorm = llama.RMSNorm(vocab_size, eps=eps)(sample)

    rmsnorm_matches = torch.allclose(orig_llama_rmsnorm, llama_rmsnorm)

    print(f"Comparing rmsnorm:\t\t{'OK' if rmsnorm_matches else 'KO'}")


@torch.no_grad()
def copy_mlp(llama_mlp, orig_llama_mlp):
    orig_llama_mlp.w1.weight.copy_(llama_mlp.c_fc1.weight)
    orig_llama_mlp.w3.weight.copy_(llama_mlp.c_fc2.weight)
    orig_llama_mlp.w2.weight.copy_(llama_mlp.c_proj.weight)


@torch.no_grad()
def copy_attention(llama_attn, orig_llama_attn):
    n_embd = llama_attn.c_attn.weight.shape[1]
    orig_llama_attn.wq.weight.copy_(llama_attn.c_attn.weight[:n_embd])
    orig_llama_attn.wk.weight.copy_(llama_attn.c_attn.weight[n_embd:-n_embd])
    orig_llama_attn.wv.weight.copy_(llama_attn.c_attn.weight[-n_embd:])
    orig_llama_attn.wo.weight.copy_(llama_attn.c_proj.weight)


@torch.no_grad()
def copy_block(llama_block, orig_llama_block):
    orig_llama_block.attention_norm.weight.copy_(llama_block.rms_1.scale)
    copy_attention(llama_block.attn, orig_llama_block.attention)
    orig_llama_block.ffn_norm.weight.copy_(llama_block.rms_2.scale)
    copy_mlp(llama_block.mlp, orig_llama_block.feed_forward)


@torch.no_grad()
def copy_weights(llama_model, orig_llama_model):
    orig_llama_model.tok_embeddings.weight.copy_(llama_model.transformer.wte.weight)
    for llama_block, orig_llama_block in zip(llama_model.transformer.h, orig_llama_model.layers):
        copy_block(llama_block, orig_llama_block)
    orig_llama_model.norm.weight.copy_(llama_model.transformer.ln_f.scale)
    orig_llama_model.output.weight.copy_(llama_model.lm_head.weight)


def compare_to_orig_llama():
    block_size = 3
    vocab_size = 32000
    n_layer = 1
    n_head = 2
    n_embd = 32

    llama_config = llama.LLaMAConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd
    )
    orig_llama_config = orig_llama.ModelArgs(
        dim=n_embd,
        n_layers=n_layer,
        n_heads=n_head,
        vocab_size=vocab_size,
        norm_eps=1e-6,
        max_seq_len=block_size
    )

    batch_size = 1

    token_sample = torch.randint(0, orig_llama_config.vocab_size, size=(batch_size, orig_llama_config.max_seq_len), dtype=torch.int64)

    llama_model = llama.LLaMA(llama_config)
    orig_llama_model = orig_llama.Transformer(orig_llama_config)

    copy_weights(llama_model, orig_llama_model)

    with torch.no_grad():
        expected = orig_llama_model(token_sample, 0)
        out = llama_model(token_sample)


    forward_matches = torch.allclose(out, expected)
    print(f"Comparing forward:\t\t{'OK' if forward_matches else 'KO'}")


@contextmanager
def on_dtype(dtype):
    original = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(original)


def compare_with_loaded_checkpoint():
    original_ckpt_path = "/srv/data/checkpoints/llama/raw/7B/consolidated.00.pth"
    ckpt_path = "/srv/data/checkpoints/llama/converted_nano/7B/state_dict.pth"

    device = torch.device("cuda")
    dtype = torch.float32

    batch_size = 1
    vocab_size = 32000
    max_seq_len = 2048
    token_sample = torch.randint(0, vocab_size, size=(batch_size, max_seq_len), dtype=torch.int64, device=device)

    seq_len = token_sample.shape[1]
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)

    print("Loading original...")
    with device:
        original_config = orig_llama.ModelArgs(dim=4096, n_layers=32, n_heads=32, vocab_size=vocab_size, max_batch_size=batch_size)  # 7B config
        with on_dtype(dtype):
            orig_llama_model = orig_llama.Transformer(original_config)
        original_checkpoint = torch.load(original_ckpt_path)
        orig_llama_model.load_state_dict(original_checkpoint, strict=False)

    orig_llama_embed = orig_llama_model.tok_embeddings(token_sample)
    orig_llama_block_out = orig_llama_model.layers[0](orig_llama_embed, 0, orig_llama_model.freqs_cis[: seq_len], mask)
    expected = orig_llama_model(token_sample, 0)

    del orig_llama_model
    del original_checkpoint
    del original_config

    print("Loading ours...")
    with device:
        config = llama.LLaMAConfig()  # 7B default
        with on_dtype(dtype):
            llama_model = llama.LLaMA(config)
        checkpoint = torch.load(ckpt_path)
        llama_model.load_state_dict(checkpoint, strict=True)

    llama_embed = llama_model.transformer.wte(token_sample)
    embed_matches = torch.allclose(orig_llama_embed, llama_embed)
    print(f"Comparing embed:\t\t{'OK' if embed_matches else 'KO'}")

    del checkpoint
    del orig_llama_embed

    llama_block_out = llama_model.transformer.h[0](llama_embed)
    block_matches = torch.allclose(orig_llama_block_out, llama_block_out)
    print(f"Comparing block out:\t\t{'OK' if block_matches else 'KO'}")

    del orig_llama_block_out
    del llama_block_out

    out = llama_model(token_sample)
    forward_matches = torch.allclose(out, expected)
    print(f"Comparing forward:\t\t{'OK' if forward_matches else 'KO'}")


if __name__ == "__main__":
    wd = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(wd)

    from scripts.download import download_original

    download_original(wd)

    import model as llama
    import original_model as orig_llama

    # compare_rope()
    #compare_rmsnorm()
    compare_to_orig_llama()
    # compare_with_loaded_checkpoint()
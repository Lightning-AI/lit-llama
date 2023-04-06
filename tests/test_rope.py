import torch


@torch.no_grad()
def test_rope(lit_llama, orig_llama) -> None:
    bs, seq_len, n_head, n_embed = 1, 6, 2, 8
    x = torch.randint(0, 10000, size=(bs, seq_len, n_head, n_embed // n_head)).float()

    freqs_cis = orig_llama.precompute_freqs_cis(n_embed // n_head, seq_len)
    llama_rope_cache = lit_llama.build_rope_cache(seq_len, n_embed // n_head, dtype=x.dtype, device=x.device)
    assert torch.equal(freqs_cis, llama_rope_cache)

    llama_x_rope = lit_llama.apply_rope(x.transpose(1, 2), llama_rope_cache).transpose(1, 2)
    orig_llama_x_rope, _ = orig_llama.apply_rotary_emb(x, x, freqs_cis)

    assert torch.equal(llama_x_rope, orig_llama_x_rope)

    # For posterity, we show here that our older implementation we initially wanted to use
    # is not numerically equivalent to Meta's rope implementation
    llama_rope_cache_old = build_rope_cache_old(seq_len, n_embed // n_head, dtype=x.dtype)
    llama_x_rope_old = apply_rope_old(x, llama_rope_cache_old)
    assert not torch.allclose(llama_x_rope_old, orig_llama_x_rope)

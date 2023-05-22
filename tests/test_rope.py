import torch


@torch.no_grad()
def test_rope(lit_llama, orig_llama) -> None:
    torch.manual_seed(1)

    bs, seq_len, n_head, n_embed = 1, 6, 2, 8
    x = torch.randint(0, 10000, size=(bs, seq_len, n_head, n_embed // n_head)).float()

    freqs_cis = orig_llama.precompute_freqs_cis(n_embed // n_head, seq_len)
    llama_rope_cache = lit_llama.build_rope_cache(seq_len, n_embed // n_head, dtype=x.dtype, device=x.device)
    torch.testing.assert_close(freqs_cis, torch.view_as_complex(llama_rope_cache))

    llama_x_rope = lit_llama.apply_rope(x, llama_rope_cache)
    orig_llama_x_rope, _ = orig_llama.apply_rotary_emb(x, x, freqs_cis)
    torch.testing.assert_close(llama_x_rope, orig_llama_x_rope)

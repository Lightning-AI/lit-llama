"""
Full definition of a LLaMA Language Model, all of it in this single file.
Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


def build_rope_cache(seq_len, n_elem, dtype, device, base=10000):
    """
    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/rope/__init__.py
    MIT License: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1. / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device, dtype=dtype)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)
    print("after outer model", idx_theta.shape, idx_theta.sum())

    # Concatenate so that for row $m$ we have
    # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

    # Cache them
    cos_cache = idx_theta2.cos()[None, None, :, :]
    sin_cache = idx_theta2.sin()[None, None, :, :]

    return torch.stack((cos_cache, sin_cache), dim=0)

# def build_rope_cache2(seq_len, n_elem, dtype, device, base=10000):
#     # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
#     theta = 1. / (base ** (torch.arange(0, n_elem, 2).float() / n_elem)).to(device)

#     # Create position indexes `[0, 1, ..., seq_len - 1]`
#     seq_idx = torch.arange(seq_len, device=device).float().to(device)

#     # Calculate the product of position index and $\theta_i$
#     idx_theta = torch.einsum('n,d->nd', seq_idx, theta)

#     # Concatenate so that for row $m$ we have
#     # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
#     idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

#     # Cache them
#     cos_cached = idx_theta2.cos()[:, None, None, :]
#     sin_cached = idx_theta2.sin()[:, None, None, :]

#     return torch.stack((cos_cached, sin_cached), dim=0)


def rotate_neg_half(x: torch.Tensor):
    # $\frac{d}{2}$
    d_2 = x.shape[-1] // 2

    # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
    return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)


def apply_rope(x: torch.Tensor, rope_cache):
    neg_half_x = rotate_neg_half(x)
    cos, sin = rope_cache

    # truncate to support variable sizes
    T = x.size(2)
    cos = cos[:, :, :T]
    sin = sin[:, :, :T]

    return (x * cos) + (neg_half_x * sin)


def apply_rope_2(x: torch.Tensor, rope_cache):

    # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
    x_rope, x_pass = x[..., :self.d], x[..., self.d:]

    # Calculate
    # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
    neg_half_x = self._neg_half(x_rope)

    # Calculate
    #
    # \begin{align}
    # \begin{pmatrix}
    # x^{(i)}_m \cos m \theta_i - x^{(i + \frac{d}{2})}_m \sin m \theta_i \\
    # x^{(i + \frac{d}{2})}_m \cos m\theta_i + x^{(i)}_m \sin m \theta_i \\
    # \end{pmatrix} \\
    # \end{align}
    #
    # for $i \in {1, 2, ..., \frac{d}{2}}$
    x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])

    #
    return torch.cat((x_rope, x_pass), dim=-1)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"{freqs_cis.shape} {x.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)



class RMSNorm(nn.Module):
    """
    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
    BSD 3-Clause License: https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE
    """

    def __init__(self, size, dim=-1, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x*x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


class CausalSelfAttention(nn.Module):

    def __init__(self, config, freqs_cis):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.freqs_cis = freqs_cis

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)

        self.q_before_rope = q

        # q = apply_rope(q, self.rope_cache)
        # k = apply_rope(k, self.rope_cache)
        print("in model", self.freqs_cis.shape)
        q, k = apply_rotary_emb(q.transpose(1, 2), k.transpose(1, 2), freqs_cis=self.freqs_cis[:T])
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        self.q_after_rope = q

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        N = 256
        # ensure n_hidden is multiple of N
        n_hidden = ((n_hidden - 1) // N) * N + N

        self.c_fc1   = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2   = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj  = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x):
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config, freqs_cis):
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, freqs_cis)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        y = self.rms_1(x)
        self.y = y
        x = x + self.attn(y)
        self.attn_x = x
        z = self.rms_2(x)
        self.z = z
        x = x + self.mlp(z)
        return x


@dataclass
class LLaMAConfig:
    block_size: int = 4096  # 7B
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096


class LLaMA(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # self.rope_cache = build_rope_cache(
        #     seq_len=config.block_size,
        #     n_elem=config.n_embd // config.n_head,
        #     dtype=self.lm_head.weight.dtype,
        #     device=self.lm_head.weight.device,
        # )
        print("inputs", config.n_embd, config.n_head, config.block_size * 2)
        self.freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_head, config.block_size * 2
        )

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config, self.freqs_cis) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))

    def forward(self, idx):
        _, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the LLaMA model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)

        self.transformer_out = x
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return logits

    def step(self, idx, targets):
        logits = self(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return loss

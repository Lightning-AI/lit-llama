"""Implementation of the paper:

LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention
https://arxiv.org/abs/2303.16199
"""
# mypy: ignore-errors
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from lit_llama.model import RMSNorm, MLP
# from lit_llama.model import build_rope_cache, apply_rope


def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device, base: int = 10000) -> torch.Tensor:
    """This is the `build_rope_cache` implementation we initially intended to use, but it is numerically not
    exactly equivalent to the one in the Meta model. We keep it here for posterity.

    Derived from:mers/rope/__init__.py
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license MIT License:
    """  # noqa: E501
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    # Concatenate so that for row $m$ we have
    # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

    # Cache them
    cos_cache = idx_theta2.cos()[None, None, :, :]
    sin_cache = idx_theta2.sin()[None, None, :, :]

    return torch.stack((cos_cache, sin_cache), dim=0)


def rotate_neg_half(x: torch.Tensor) -> torch.Tensor:
    # $\frac{d}{2}$
    d_2 = x.shape[-1] // 2
    # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$  # noqa: E501
    return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """This is the `apply_rope` implementation we initially intended to use, but it is numerically not exactly
    equivalent to the one in the Meta model.

    We keep it here for posterity.
    """
    neg_half_x = rotate_neg_half(x)
    cos, sin = rope_cache
    # truncate to support variable sizes
    T = x.size(2)
    cos = cos[:, :, :T]
    sin = sin[:, :, :T]
    return (x * cos) + (neg_half_x * sin)


@dataclass
class LLaMAConfig:
    # Default configuration is the 7B model
    block_size: int = 4096
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096

    adapter_prompt_length: int = 10
    adapter_start_layer: int = 2


class CausalSelfAttention(nn.Module):
    """A modification of `lit_llama.model.CausalSelfAttention` that adds the attention
    over the adaption prompt."""

    def __init__(self, config: LLaMAConfig, block_idx: int) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        if block_idx >= config.adapter_start_layer:
            # adapter embedding layer
            self.adapter_wte = nn.Embedding(config.adapter_prompt_length, config.n_embd)
            # gate for adaption
            self.gating_factor = torch.nn.Parameter(torch.ones(1))

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.block_idx = block_idx
        self.adapter_prompt_length = config.adapter_prompt_length
        self.adapter_start_layer = config.adapter_start_layer
        self.rope_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)  # (B, nh, T, hs)

        if self.rope_cache is None:
            # cache for future forward calls
            self.rope_cache = build_rope_cache(
                seq_len=self.block_size,
                n_elem=self.n_embd // self.n_head, 
                dtype=self.c_attn.weight.dtype,
                device=x.device,
            )

        q = apply_rope(q, self.rope_cache)
        k = apply_rope(k, self.rope_cache)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #  att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #  att = F.softmax(att, dim=-1)
        #  y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        if self.block_idx >= self.adapter_start_layer:
            prefix = self.adapter_wte.weight.reshape(1, self.adapter_prompt_length, self.n_embd)

            # inefficient attention because we need to insert the gate for the adaption in the middle
            aT = prefix.size(1)
            _, ak, av = self.c_attn(prefix).split(self.n_embd, dim=2)
            ak = ak.view(1, aT, self.n_head, head_size).repeat(B, 1, 1, 1).transpose(1, 2)
            av = av.view(1, aT, self.n_head, head_size).repeat(B, 1, 1, 1).transpose(1, 2)

            print("our ak", ak.contiguous().view(-1)[:5].tolist())
            print("our av", av.contiguous().view(-1)[:5].tolist())

            print("our xq", x.contiguous().view(-1)[:5].tolist())

            ascores = torch.matmul(q, ak.transpose(2, 3)) / math.sqrt(self.n_embd)
            print("our ascores", ascores.contiguous().view(-1)[:5].tolist())
            ascores = self.gating_factor * F.softmax(ascores.float(), dim=-1).type_as(q)
            y = y + torch.matmul(ascores, av)
           

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y


class Block(nn.Module):
    """The implementation is identical to `lit_llama.model.Block` with the exception that
    we replace the attention layer where adaption is implemented."""

    def __init__(self, config: LLaMAConfig, block_idx: int) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, block_idx)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x))
        x = x + self.mlp(self.rms_2(x))
        return x


class LLaMA(nn.Module):
    """The implementation is identical to `lit_llama.model.LLaMA` with the exception that
    the `Block` saves the layer index and passes it down to the attention layer."""

    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the LLaMA model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits



def mark_only_adapter_as_trainable(model: LLaMA) -> None:
    """Sets `requires_grad=False` for all non-adapter weights."""
    for name, param in model.named_parameters():
        param.requires_grad = "adapter_wte" in name or "gating_factor" in name


def adapter_state_dict(model: LLaMA) -> dict:
    """Retrieve the model state dict with only the adapter weights for saving."""
    return {
        name: param for name, param in model.named_parameters() if "adapter_wte" in name or "gating_factor" in name
    }

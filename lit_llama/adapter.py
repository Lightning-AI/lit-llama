"""Implementation of the paper:

LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention
https://arxiv.org/abs/2303.16199

                                                                             |              Prefix cross-attention
                                                                             |
  ┌─────────────────┐                                                        |               ┌──────────────────┐
  ┆        x        ┆                                                        |               ┆      prefix      ┆
  └─────────────────┘                                                        |               └──────────────────┘
           |                                                                 |                        |
           ▼                                                                 |                        ▼
  ┌──────────────────┐                                                       |              ┌─────────────────────┐
  ┆  self-attention  ┆ --------------------------------------------------------------┐      ┆  linear projection  ┆
  └──────────────────┘                                                       |       ┆      └─────────────────────┘
           |                                                                 |       ┆                |         \
           ▼                                                                 |       ▼                ▼          ▼
         ╭───╮     ┌────────────────┐ ╭───╮ ┌──────────────────────────┐     |  ┌─────────┐    ┌──────────────┐  ┌────────────────┐
         ┆ + ┆ ◀── ┆  gating factor ┆-┆ x ┆-┆  prefix cross-attention  ┆     |  ┆  query  ┆    ┆  prefix key  ┆  ┆  prefix value  ┆
         ╰───╯     └────────────────┘ ╰───╯ └──────────────────────────┘     |  └─────────┘    └──────────────┘  └────────────────┘
           |                                                                 |          \             |           /
           ▼                                                                 |           ▼            ▼          ▼
                                                                             |         ┌────────────────────────────────┐
                                                                             |         ┆  scaled dot-product attention  ┆
                                                                             |         └────────────────────────────────┘


In order to inject learnable information from the prefix to pretrained weights we need to sum outputs from
self-attention and prefix cross-attention (times gating factor). For prefix cross-attention we need `query` (from
self-attention as a result of linear projection), `prefix key` and `prefix value` (from cross-attention as a result of
linear projection).
The output of prefix cross-attention is multiplied by gating factor, which is a learnable parameter that is needed to
avoid potential disruption of pretrained weights caused by incorporating randomly initialized tensors. This factor is
initialized with zeros to avoid noise from the adaption prompts at the early training stage.
More about it: https://lightning.ai/pages/community/article/understanding-llama-adapters/

Notes about implementation: as per paper adapter's prefix is concatenated with the input, while here outputs of
self-attention and prefix cross-attention are summed. Both variants are mathematically equivalent:
https://github.com/ZrrSkywalker/LLaMA-Adapter/issues/47
"""
# mypy: ignore-errors
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

import lit_llama.model as llama
from lit_llama.model import build_rope_cache, apply_rope, RMSNorm, MLP, KVCache, RoPECache


@dataclass
class LLaMAConfig(llama.LLaMAConfig):
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
            # a learnable gating factor (to avoid potential disruption of pretrained weights) initialized with zeros (to
            # avoid noise from adaption prompts at the early training stage)
            self.gating_factor = torch.nn.Parameter(torch.zeros(1, config.n_head, 1, 1))

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.block_idx = block_idx
        self.adapter_prompt_length = config.adapter_prompt_length
        self.adapter_start_layer = config.adapter_start_layer

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: torch.Tensor,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        adapter_kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache], Optional[KVCache]]:
        # notation:
        # - B  | batch
        # - T  | time-step (sequence length)
        # - C  | embeddings size (n_embd) = head size * num heads
        # - hs | head size
        # - nh | number of heads

        B, T, C = x.size()

        # instead of calculating `query`, `key` and `value` by separately multiplying input `x` with corresponding
        # weight matrices do it (for all heads) in a single multiplication with a matrix of 3x size (concatenated
        # weights for q, k, v) and then split the result along `embedding size` dimension
        q, k, v = self.c_attn(x).split(C, dim=2) # (B, T, 3 * C) --> 3 * (B, T, C)

        # in order to move head_size (hs) dimension right after batch (B) dimension, we need to first split
        # embedding size (C) dimension into num_heads (nh) and head_size (hs)
        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size)
        q = q.view(B, T, self.n_head, head_size)
        v = v.view(B, T, self.n_head, head_size)

        # "Unlike standard positional embeddings rotary embeddings must be applied at every layer"
        q = apply_rope(q, rope) # (B, T, nh, hs)
        k = apply_rope(k, rope) # (B, T, nh, hs)

        # now `key`, 'query` and `value` tensors are correctly represented: for each element in a batch (B)
        # there is a number of heads (nh) and for each head there is a sequence of elements (T), each of them is
        # represented by a vector of size `hs`
        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache # 2 * (B, nh, max_seq_length, hs)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                # if we reached token limit and thus there is no space to put newly calculated `key` and `value`
                # right next to cached ones, we need to rotate cache tensor along `max_seq_length` dimension by one
                # element to the left: this will free up space for new `key` and `value`
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
            k = cache_k.index_copy(2, input_pos, k) # (B, nh, max_seq_length, hs)
            v = cache_v.index_copy(2, input_pos, v) # (B, nh, max_seq_length, hs)
            kv_cache = k, v

        # efficient attention using Flash Attention CUDA kernels
        # ↓ (B, nh, T, hs) @ (B, nh, T, hs).mT --> (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0) # (B, nh, T, hs)

        # "Adapters are applied to the topmost layers to better tune the language
        # representations with higher-level semantics".
        if self.block_idx >= self.adapter_start_layer:
            aT = self.adapter_prompt_length
            if adapter_kv_cache is not None:
                ak, av = adapter_kv_cache # 2 * (B, nh, aT, hs)
            else:
                prefix = self.adapter_wte.weight.reshape(1, aT, C)
                _, ak, av = self.c_attn(prefix).split(C, dim=2) # (1, aT, 3 * C) --> 3 * (1, aT, C)
                ak = ak.view(1, aT, self.n_head, head_size).repeat(B, 1, 1, 1).transpose(1, 2) # (B, nh, aT, hs)
                av = av.view(1, aT, self.n_head, head_size).repeat(B, 1, 1, 1).transpose(1, 2) # (B, nh, aT, hs)
                adapter_kv_cache = (ak, av)

            # Apply cross-attention with `query`, `adapter_key`, `adapter_value` and sum the output with the output
            # obtained from self-attention step. This is mathematically equivalent to concatenation of prefix and input as per paper.
            amask = torch.ones(T, aT, dtype=torch.bool, device=x.device)
            # ↓ (B, nh, T, hs) @ (B, nh, aT, hs).mT --> (B, nh, T, aT) @ (B, nh, aT, hs) --> (B, nh, T, hs)
            ay = F.scaled_dot_product_attention(q, ak, av, attn_mask=amask, dropout_p=0.0, is_causal=False) # (B, nh, T, hs)
            y = y + self.gating_factor * ay

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y) # (B, T, C)

        return y, kv_cache, adapter_kv_cache

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """For backward compatibility with old checkpoints that have a single gating value for all heads."""
        name = prefix + "gating_factor"
        if name in state_dict:
            tensor = state_dict[name]
            # in case we are loading with `utils.lazy_load()`
            tensor = tensor._load_tensor() if hasattr(tensor, "_load_tensor") else tensor

            if len(tensor.shape) < 4:
                # For old checkpoints with unified gating value
                state_dict[name] = tensor.reshape(1, 1, 1, 1).repeat(1, self.n_head, 1, 1)
            else:
                state_dict[name] = tensor

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Block(nn.Module):
    """The implementation is identical to `lit_llama.model.Block` with the exception that
    we replace the attention layer where adaption is implemented."""

    def __init__(self, config: LLaMAConfig, block_idx: int) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, block_idx)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: torch.Tensor,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        adapter_kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache], Optional[KVCache]]:
        h, new_kv_cache, new_adapter_kv_cache = self.attn(
            self.rms_1(x), rope, mask, max_seq_length, input_pos, kv_cache, adapter_kv_cache
        )
        x = x + h
        x = x + self.mlp(self.rms_2(x))
        return x, new_kv_cache, new_adapter_kv_cache


class LLaMA(llama.LLaMA):
    """The implementation is identical to `lit_llama.model.LLaMA` with the exception that
    the `Block` saves the layer index and passes it down to the attention layer."""

    def __init__(self, config: LLaMAConfig) -> None:
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
                ln_f=RMSNorm(config.n_embd),
            )
        )

        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []
        self.adapter_kv_caches: List[KVCache] = []

    @classmethod
    def from_name(cls, name: str):
        return cls(LLaMAConfig.from_name(name))

    def reset_cache(self) -> None:
        super().reset_cache()
        self.adapter_kv_caches.clear()

    def forward(
        self, idx: torch.Tensor, max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[KVCache]]]:
        B, T = idx.size()

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        assert T <= max_seq_length, f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert T <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx) # (block_size, head_size / 2, 2)
        if self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx) # (1, 1, block_size, block_size)

        if input_pos is not None:
            rope = self.rope_cache.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            rope = self.rope_cache[:T]
            mask = self.mask_cache[:, :, :T, :T]

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)

        if input_pos is None:  # proxy for use_cache=False
            for block in self.transformer.h:
                x, *_ = block(x, rope, mask, max_seq_length)
        else:
            if not self.kv_caches:
                head_size = self.config.n_embd // self.config.n_head
                cache_shape = (B, self.config.n_head, max_seq_length, head_size)
                self.kv_caches = [
                    (torch.zeros(cache_shape, device=x.device, dtype=x.dtype), torch.zeros(cache_shape, device=x.device, dtype=x.dtype))
                    for _ in range(self.config.n_layer)
                ]
            if not self.adapter_kv_caches:
                self.adapter_kv_caches = [None for _ in range(self.config.n_layer)]
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i], self.adapter_kv_caches[i] = block(
                    x, rope, mask, max_seq_length, input_pos, self.kv_caches[i], self.adapter_kv_caches[i]
                )

        x = self.transformer.ln_f(x) # (B, T, n_embd)

        logits = self.lm_head(x)  # (B, T, padded_vocab_size)

        return logits


def mark_only_adapter_as_trainable(model: LLaMA) -> None:
    """Sets `requires_grad=False` for all non-adapter weights."""
    for name, param in model.named_parameters():
        param.requires_grad = "adapter_wte" in name or "gating_factor" in name


def adapter_state_from_state_dict(state_dict: dict) -> dict:
    """Returns the model state dict with only the adapter weights for saving."""
    return {name: param for name, param in state_dict.items() if "adapter_wte" in name or "gating_factor" in name}

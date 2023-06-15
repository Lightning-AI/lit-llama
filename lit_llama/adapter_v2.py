import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

from lit_llama.utils import find_multiple

from sentence_transformers import SentenceTransformer, util
from PIL import Image
import clip
from timm.models.vision_transformer import Block as ViTBlock
import re


@dataclass
class LLaMAConfig:
    block_size: int = 2048
    vocab_size: int = 32000
    padded_vocab_size: Optional[int] = None
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    adapter_prompt_length: int = 10
    adapter_start_layer: int = 2

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, 64)

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**llama_configs[name])


llama_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096),
    "13B": dict(n_layer=40, n_head=40, n_embd=5120),
    "30B": dict(n_layer=60, n_head=52, n_embd=6656),
    "65B": dict(n_layer=80, n_head=64, n_embd=8192),
}


class LLaMA(nn.Module):
    """The implementation is identical to `lit_llama.model.LLaMA` with the exception that
    the `Block` saves the layer index and passes it down to the attention layer."""

    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Note: In the paper they apply bias tuning to all layers, but we omit the output layer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )

        v_embed_dim = 768
        v_depth = 8
        v_num_heads = 16

        clip_dim = 768
        self.clip_proj = nn.Linear(clip_dim, v_embed_dim)
        self.clip_proj_norm = nn.LayerNorm(v_embed_dim)

        # 2. visual query, blocks and projector
        self.visual_query = nn.Embedding(config.adapter_prompt_length, v_embed_dim)
        self.visual_blocks = nn.ModuleList([ViTBlock(v_embed_dim, v_num_heads, mlp_ratio=4.0, qkv_bias=True) for _ in range(v_depth)])
        self.visual_proj = nn.Linear(v_embed_dim, config.n_embd)
        self.visual_proj_norm = nn.LayerNorm(config.n_embd)


    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(self, idx, img_features=None) -> torch.Tensor:
        _, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        
        visual_context = self.forward_visual(img_features) if img_features is not None else 0

        for block in self.transformer.h:
            x = block(x, visual_context)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits

    def clip_encode_image(self, x):
        # modified from CLIP
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x

    def forward_visual(self, img_features):
        # clip_feats = self.clip_encode_image(imgs)
        batch_size = len(img_features)
        clip_feats = self.clip_proj_norm(self.clip_proj(img_features.to(self.clip_proj.weight.dtype)))

        visual_query = self.visual_query.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        visual_query = torch.cat([visual_query, clip_feats], dim=1)
        for block in self.visual_blocks:
            visual_query = block(visual_query)

        visual_query = visual_query[:, :self.config.adapter_prompt_length, :]
        visual_query = self.visual_proj(visual_query)
        visual_query = self.visual_proj_norm(visual_query)

        return visual_query

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(LLaMAConfig.from_name(name))


class Block(nn.Module):
    """The implementation is identical to `lit_llama.model.Block` with the exception that
    we replace the attention layer where adaption is implemented."""

    def __init__(self, config: LLaMAConfig, block_idx: int) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, block_idx)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, visual_context) -> torch.Tensor:
        x = x + self.attn(self.rms_1(x), visual_context)
        x = x + self.mlp(self.rms_2(x))
        return x


class CausalSelfAttention(nn.Module):
    """A modification of `lit_llama.model.CausalSelfAttention` that adds the attention
    over the adaption prompt."""

    def __init__(self, config: LLaMAConfig, block_idx: int) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = AdapterLinear(config.n_embd, 3 * config.n_embd)
        self.c_proj = AdapterLinear(config.n_embd, config.n_embd)
        
        if block_idx >= config.adapter_start_layer:
            # adapter embedding layer
            self.adapter_wte = nn.Embedding(config.adapter_prompt_length, config.n_embd)
            # gate for adaption
            self.gating_factor = torch.nn.Parameter(torch.zeros(1))

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.block_idx = block_idx
        self.adapter_prompt_length = config.adapter_prompt_length
        self.adapter_start_layer = config.adapter_start_layer
        self.rope_cache = None

    def forward(self, x: torch.Tensor, visual_context) -> torch.Tensor:
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
                dtype=x.dtype,
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
            # print("shapes", prefix.shape, visual_context.shape)
            prefix = prefix + visual_context
            # print("prefix", prefix.shape)
            # if isinstance(visual_context, torch.Tensor):
            #     print("visual", visual_context.shape)

            # print("batch size", B)
            prefix = prefix.expand(B, -1, -1)
            # print("prefix after expand", prefix.shape)

            

            aT = prefix.size(1)
            _, ak, av = self.c_attn(prefix).split(self.n_embd, dim=2)
            ak = ak.view(-1, aT, self.n_head, head_size).transpose(1, 2)
            av = av.view(-1, aT, self.n_head, head_size).transpose(1, 2)

            amask = torch.ones(q.shape[-2], ak.shape[-2], dtype=torch.bool, device=x.device)
            ay = F.scaled_dot_product_attention(q, ak, av, attn_mask=amask, dropout_p=0.0, is_causal=False)
            y = y + self.gating_factor * ay

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)

        self.c_fc1 = AdapterLinear(config.n_embd, n_hidden)
        self.c_fc2 = AdapterLinear(config.n_embd, n_hidden)
        self.c_proj = AdapterLinear(n_hidden, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


def build_rope_cache(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000) -> torch.Tensor:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    x = x.transpose(1, 2)

    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
         xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ], -1)

    x_out2 = x_out2.flatten(3)
    return x_out2.transpose(1, 2).type_as(x)
    


class AdapterLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=True, device=device, dtype=dtype)
        nn.init.constant_(self.bias.data, 0)
        # Note: Paper also adds a scale factor, but we don't add one


INSTRUCTION_ADAPTER_REGEX = ".*transformer.*adapter_wte|.*transformer.*gating_factor|.*transformer.*bias|.*rms_1.*|.*rms_2.*|.*ln_f.*"
VISUAL_ADAPTER_REGEX = ".*clip_proj.*|.*visual_proj.*|.*visual_blocks.*|.*visual_query.*"


def mark_instruction_adapter_trainable(model: LLaMA) -> None:
    """Sets `requires_grad=False` for all parameters except late adaptation prompts, zero-init gating,
    rms-norm, biases and scale factors."""
    for name, param in model.named_parameters():
        param.requires_grad = bool(re.match(INSTRUCTION_ADAPTER_REGEX, name))


def mark_visual_adapter_trainable(model: LLaMA) -> None:
    """Sets `requires_grad=False` for all parameters except the projection layers and 
    early zero-init attention with gating."""
    for name, param in model.named_parameters():
        param.requires_grad = bool(re.match(VISUAL_ADAPTER_REGEX, name))


def adapter_state_from_state_dict(state_dict: dict) -> dict:
    """Returns the model state dict with only the adapter weights for saving."""
    return {
        name: param 
        for name, param in state_dict.items()
        if re.match(INSTRUCTION_ADAPTER_REGEX, name) or re.match(VISUAL_ADAPTER_REGEX, name)
    }

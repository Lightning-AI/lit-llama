"""
Utility functions to extend the original LLaMA-Adapter method to LLaMA-Adapter v2,
based on the paper,
LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model
https://arxiv.org/abs/2304.15010
"""
import torch
from torch import Tensor
from torch.nn import functional as F

from lit_llama.adapter import LLaMA


def adapter_v2_mark_only_adapter_as_trainable(model: LLaMA) -> None:
    """Sets `requires_grad=False` for all non-adapter weights."""
    for name, param in model.named_parameters():
        substrings = ("adapter_wte", "gating_factor", "adapter_scale", "adapter_bias", "RMSNorm")
        if "RMSNorm" in name:
            print(name)
        param.requires_grad = any(s in name for s in substrings)


def adapter_v2_state_from_state_dict(state_dict: dict) -> dict:
    """Returns the model state dict with only the adapter weights for saving."""
    substrings = ("adapter_wte", "gating_factor", "adapter_scale", "adapter_bias")
    return {name: param for name, param in state_dict.items() if any(s in name for s in substrings)}


def adapter_v2_new_forward(self, input: Tensor) -> Tensor:
    return self.adapter_scale * (
        F.linear(input, self.weight, self.bias) + self.adapter_bias
    )


def adapter_v2_linear_with_bias_and_scale(layer):
    layer.adapter_bias = torch.nn.Parameter(torch.zeros(layer.weight.shape[0]), requires_grad=True)
    layer.adapter_scale = torch.nn.Parameter(torch.ones(layer.weight.shape[0]), requires_grad=True)
    bound_method = adapter_v2_new_forward.__get__(layer, layer.__class__)
    setattr(layer, 'forward', bound_method)
    return layer

import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D


class MergedLinear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        if out_features % len(enable_lora) != 0:
            raise ValueError("The length of enable_lora must divide out_features")
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Linear(in_features, r * sum(enable_lora), bias=False)
            self.lora_B = nn.Conv1d(
                r * sum(enable_lora),
                out_features // len(enable_lora) * sum(enable_lora),
                kernel_size=1,
                groups=2,
                bias=False,
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        if not mode and self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_w = (
                    F.conv1d(
                        self.lora_A.weight.data.unsqueeze(0),
                        self.lora_B.weight.data,
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .transpose(-2, -1)
                )
                self.weight.data += transpose(
                    self.zero_pad(delta_w * self.scaling), not self.fan_in_fan_out
                )
            self.merged = True
        elif self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_w = (
                    F.conv1d(
                        self.lora_A.weight.data.unsqueeze(0),
                        self.lora_B.weight.data,
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .transpose(-2, -1)
                )
                self.weight.data -= transpose(
                    self.zero_pad(delta_w * self.scaling), not self.fan_in_fan_out
                )
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r > 0 and self.merged and any(self.enable_lora):
                delta_w = (
                    F.conv1d(
                        self.lora_A.weight.data.unsqueeze(0),
                        self.lora_B.weight.data,
                        groups=sum(self.enable_lora),
                    )
                    .squeeze(0)
                    .transpose(-2, -1)
                )
                self.weight.data -= transpose(
                    self.zero_pad(delta_w * self.scaling), not self.fan_in_fan_out
                )
                self.merged = False
            return F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        elif self.merged:
            return F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        else:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
            if self.r > 0:
                after_A = self.lora_A(self.lora_dropout(x))
                after_B = self.lora_B(after_A.transpose(-2, -1)).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result

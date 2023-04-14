"""Utility functions for training and inference."""

import functools
from pathlib import Path

import torch
import torch.utils._device
from lightning.fabric.strategies import DeepSpeedStrategy, FSDPStrategy
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType


def save_model_checkpoint(fabric, model, file_path):
    """Handles boilerplate logic for retrieving and saving the state_dict.
    
    This will be upstreamed to Fabric soon.
    """
    file_path = Path(file_path)

    if isinstance(fabric.strategy, DeepSpeedStrategy):
        from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

        fabric.save(file_path, {"model": model})
        fabric.barrier()
        if fabric.global_rank == 0:
            # Create a consolidated checkpoint with the same name next to the deepspeed checkpoint
            convert_zero_checkpoint_to_fp32_state_dict(file_path, file_path.with_suffix(".pt"))
        return

    if isinstance(fabric.strategy, FSDPStrategy):
        save_policy = FullStateDictConfig(offload_to_cpu=(fabric.world_size > 1), rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model._forward_module.state_dict()
    else:
        state_dict = model.state_dict()

    if fabric.global_rank == 0:
        torch.save(state_dict, file_path)
    fabric.barrier()


class EmptyInitOnDevice(torch.overrides.TorchFunctionMode):
    def __init__(self, device=None, dtype=None, quantization_mode=None):
        """
        Create tensors with given device and dtype and don't run initialization
           (but instead use "empty tensors", i.e. uninitialized memory).

            device: `torch.device` to work with
            dtype: `torch.dtype` to work with
            quantization_mode: optional string, quantization mode to work with, default `None`.
                 Available modes: `llm.int8` bitsnbytes LLM.int8 quantization (only on GPU)

        Example::
            with EmptyInitOnDevice("cuda", dtype=torch.bfloat16):
               model = LLaMA.from_name('7B')
            model.load_state_dict(torch.load('llama-lit/7B/state_dict.pth'))"""

        self.quantization_mode = quantization_mode
        self.quantized_linear_cls = None
        if self.quantization_mode == 'llm.int8':
            if device.type != "cuda":
                raise ValueError("Quantization is only supported on the GPU.")
            from .quantization import Linear8bitLt
            self.quantized_linear_cls = Linear8bitLt
        elif self.quantization_mode == 'gptq.int4':
            from .quantization import ColBlockQuantizedLinear
            self.quantized_linear_cls = functools.partial(ColBlockQuantizedLinear, bits=4, tile_cols=-1)
        elif self.quantization_mode == 'gptq.int8':
            from .quantization import ColBlockQuantizedLinear
            self.quantized_linear_cls = functools.partial(ColBlockQuantizedLinear, bits=8, tile_cols=-1)
        elif self.quantization_mode is not None:
            raise RuntimeError(f"unknown quantization mode {self.quantization_mode}")
        self.device = device
        self.dtype = dtype

    def __enter__(self):
        if self.quantized_linear_cls != None:
            self.torch_linear_cls = torch.nn.Linear
            torch.nn.Linear = self.quantized_linear_cls
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.quantized_linear_cls != None:
            torch.nn.Linear = self.torch_linear_cls
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, "__module__", None) == "torch.nn.init":
            if "tensor" in kwargs:
                return kwargs["tensor"]
            else:
                return args[0]
        if (
            self.device is not None
            and func in torch.utils._device._device_constructors()
            and kwargs.get("device") is None
        ):
            kwargs["device"] = self.device
        if (
            self.dtype is not None
            and func in torch.utils._device._device_constructors()
            and kwargs.get("dtype") is None
        ):
            kwargs["dtype"] = self.dtype
        return func(*args, **kwargs)



def build_rope_cache_complex(seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000) -> torch.Tensor:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.

    Note: This version in the complex domain is only used to match the implementation of Meta's LLaMA version
    in order to perform inference using their checkpoints.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    # Compute cache. Because polar only takes float32 or float64, we need to cast
    # when working with 16 bit floats (float16 or bfloat16)
    working_dtype = (
        torch.float32 if (dtype == torch.float16 or dtype == torch.bfloat16) else dtype
    )
    complex_dtype = (
        torch.complex32
        if (dtype == torch.float16 or dtype == torch.bfloat16)
        else torch.complex64
    )
    cache = torch.polar(
        torch.ones_like(idx_theta).to(working_dtype), idx_theta.to(working_dtype)
    ).to(complex_dtype)
    return cache


def apply_rope_complex(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    x = x.transpose(1, 2)

    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because `view_as_complex` does not support 16 bit tensors
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rope_cache = rope_cache.view(1, xc.size(1), 1, xc.size(3))
    x_out = torch.view_as_real(xc * rope_cache).flatten(3)
    return x_out.transpose(1, 2).type_as(x)

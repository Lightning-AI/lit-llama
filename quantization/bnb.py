import os
from typing import Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
import bitsandbytes as bnb  # noqa: E402



# class Int8Params(torch.nn.Parameter):
#     def __new__(
#         cls,
#         data=None,
#         requires_grad=True,
#         has_fp16_weights=False,
#         CB=None,
#         SCB=None,
#     ):
#         cls.has_fp16_weights = has_fp16_weights
#         cls.CB = None
#         cls.SCB = None
#         if data is None:
#             data = torch.empty(0)
#         return torch.Tensor._make_subclass(cls, data, requires_grad)

#     def cuda(self, device):
#         if self.has_fp16_weights:
#             return super().cuda(device)
#         else:
#             # we store the 8-bit rows-major weight
#             # we convert this weight to the turning/ampere weight during the first inference pass
#             B = self.data.contiguous().half().cuda(device)
#             CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
#             del CBt
#             del SCBt
#             self.data = CB
#             setattr(self, "CB", CB)
#             setattr(self, "SCB", SCB)

#         return self


#     def to(self, *args, **kwargs):
#         device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
#             *args, **kwargs
#         )

#         if (
#             device is not None
#             and device.type == "cuda"
#             and self.data.device.type == "cpu"
#         ):
#             return self.cuda(device)
#         else:
#             new_param = Int8Params(
#                 super().to(
#                     device=device, dtype=dtype, non_blocking=non_blocking
#                 ),
#                 requires_grad=self.requires_grad,
#                 has_fp16_weights=self.has_fp16_weights,
#             )
#             new_param.CB = self.CB
#             new_param.SCB = self.SCB

#             return new_param


# class MainLinear8bitLt(nn.Linear):
#     def __init__(self, input_features, output_features, bias=True, has_fp16_weights=True,
#                        memory_efficient_backward=False, threshold=0.0, index=None):
#         super().__init__(input_features, output_features, bias)
#         assert not memory_efficient_backward, "memory_efficient_backward is no longer required and the argument is deprecated in 0.37.0 and will be removed in 0.39.0"
#         self.state = bnb.MatmulLtState()
#         self.index = index

#         self.state.threshold = threshold
#         self.state.has_fp16_weights = has_fp16_weights
#         self.state.memory_efficient_backward = memory_efficient_backward
#         if threshold > 0.0 and not has_fp16_weights:
#             self.state.use_pool = True

#         self.weight = Int8Params(self.weight.data, has_fp16_weights=has_fp16_weights, requires_grad=has_fp16_weights)

#     def init_8bit_state(self):
#         self.state.CB = self.weight.CB
#         self.state.SCB = self.weight.SCB
#         self.weight.CB = None
#         self.weight.SCB = None

#     def forward(self, x: torch.Tensor):
#         self.state.is_training = self.training
#         if self.weight.CB is not None:
#             self.init_8bit_state()

#         # weights are cast automatically as Int8Params, but the bias has to be cast manually
#         if self.bias is not None and self.bias.dtype != x.dtype:
#             self.bias.data = self.bias.data.to(x.dtype)

#         out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)
#         if not self.state.has_fp16_weights:
#             if self.state.CB is not None and self.state.CxB is not None:
#                 # we converted 8-bit row major to turing/ampere format in the first inference pass
#                 # we no longer need the row-major weight
#                 del self.state.CB
#                 self.weight.data = self.state.CxB
#         return out



class Linear8bitLt(bnb.nn.Linear8bitLt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, has_fp16_weights=False, threshold=6.0)
        self.quantize_weight(self.weight)

    def _load_from_state_dict(self, local_state_dict, *args, **kwargs):
        weight = local_state_dict[next(iter(local_state_dict.keys()))]
        # print(weight.view(-1)[0])
        self.quantize_weight(weight)
    
    def quantize_weight(self, weight):
        B = weight.data.contiguous().half().cuda()
        CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
        del CBt
        del SCBt
        self.weight.data = CB
        setattr(self.weight, "CB", CB)
        setattr(self.weight, "SCB", SCB)
        self.init_8bit_state()
        #return super().load_state_dict(state_dict, strict)

    # def register_parameter(self, name, param):
    #     super().register_parameter(name=name, param=param)
    #     if isinstance(param, bnb.nn.Int8Params):
    #         device = param.device
    #         if device == "cuda":
    #             self.cuda(device)

    # def forward(self, x):
    #     assert x.device.type == "cuda"
    #     assert self.weight.device.type == "cuda"
    #     # assert self.bias.device.type == "cuda"
    #     return super().forward(x)


# def trigger_quantization(model):
#     for module in model.modules():
#         if isinstance(module, Linear8bitLt):
#             module.init_8bit_state()

@contextmanager
def as_8_bit_quantized(device: torch.device, enabled: bool = True):
    """A context manager under which you can instantiate the model with 8-bit quantized tensors
    being created directly on the given device."""
    if not enabled:
        yield
        return

    with torch.device(device):
        torch_linear_cls = torch.nn.Linear
        torch.nn.Linear = Linear8bitLt
        yield
        torch.nn.Linear = torch_linear_cls

# def quantize(model: nn.Module, threshold: float = 6.0, skip: Tuple[str, ...] = ()) -> nn.Module:
#     for name, module in model.named_children():
#         if isinstance(module, nn.Linear) and name not in skip:
#             linear8bit = bnb.nn.Linear8bitLt(
#                 module.in_features, module.out_features, bias=module.bias, has_fp16_weights=False, threshold=threshold
#             )

#             device = linear8bit.weight.device
#             if device == "cuda":
#                 linear8bit.cuda(device)

#         if module.children():
#             quantize(module, threshold=threshold, skip=skip)
#     return model

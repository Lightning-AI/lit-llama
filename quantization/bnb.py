import os
from typing import Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
import bitsandbytes as bnb  # noqa: E402


class Linear8bitLt(bnb.nn.Linear8bitLt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, has_fp16_weights=False, threshold=6.0)
        self.cuda()
        # self.weight.data = torch.empty_like(self.weight.data)
        # # we store the 8-bit rows-major weight
        # # we convert this weight to the turning/ampere weight during the first inference pass
        # B = self.weight.data.contiguous().half().cuda()
        # CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
        # del CBt
        # del SCBt
        # self.weight.data = CB
        # setattr(self.weight, "CB", CB)
        # setattr(self.weight, "SCB", SCB)

    def load_state_dict(self, state_dict, strict):
        # we store the 8-bit rows-major weight
        # we convert this weight to the turning/ampere weight during the first inference pass
        weight = state_dict["weight"]
        B = weight.data.contiguous().half().cuda()
        CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
        del CBt
        del SCBt
        self.weight.data = CB
        setattr(self.weight, "CB", CB)
        setattr(self.weight, "SCB", SCB)
        #return super().load_state_dict(state_dict, strict)

    # def register_parameter(self, name, param):
    #     super().register_parameter(name=name, param=param)
    #     if isinstance(param, bnb.nn.Int8Params):
    #         device = param.device
    #         if device == "cuda":
    #             self.cuda(device)


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

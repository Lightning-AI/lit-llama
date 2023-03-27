import bitsandbytes as bnb
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layer = bnb.nn.Linear8bitLt(500, 500, has_fp16_weights=False)

with torch.device("cuda"):
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Model()
    print(torch.cuda.max_memory_reserved() / 1e9)
    print(model.layer.weight.dtype)
    print(model.layer.weight.device)



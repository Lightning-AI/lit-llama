# PRs:
# https://github.com/Lightning-AI/lightning/pull/17462
# https://github.com/Lightning-AI/lightning/pull/17287

import lightning as L
import torch

from lit_llama.model import LLaMA, LLaMAConfig
torch.set_float32_matmul_precision("high")

checkpoint = torch.load("checkpoints/lit-llama/7B/lit-llama.pth")

# ----------
# 1)

# fabric = L.Fabric(devices=1)
# model = LLaMA.from_name("7B").to(fabric.device).bfloat16()
# model.load_state_dict(checkpoint, strict=False)
# print(torch.cuda.max_memory_allocated() // 1e6)  # 26953.0

# ----------
# 2)

# fabric = L.Fabric(devices=1)
# with fabric.device:
#     torch.set_default_tensor_type(torch.HalfTensor)
#     model = LLaMA.from_name("7B").bfloat16()
#     torch.set_default_tensor_type(torch.FloatTensor)
#     model.load_state_dict(checkpoint, strict=False)
# print(torch.cuda.max_memory_allocated() // 1e6)  # 13738.0


# ----------
# 3)

# fabric = L.Fabric(devices=1, precision="bf16-true")
# with fabric.init_module():
#     model = LLaMA.from_name("7B")
#     model.load_state_dict(checkpoint, strict=False)
# print(torch.cuda.max_memory_allocated() // 1e6)


# ----------
# Future

# from lit_llama.utils import EmptyInitOnDevice

# fabric = L.Fabric(devices=1)
# with EmptyInitOnDevice(device=fabric.device, dtype=torch.bfloat16):
#     model = LLaMA.from_name("7B")
#     model.load_state_dict(checkpoint, strict=False)
# print(torch.cuda.max_memory_allocated() // 1e6)

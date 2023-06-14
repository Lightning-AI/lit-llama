import re

pattern = ".*transformer.*adapter_wte|.*transformer.*gating_factor|.*transformer.*bias|.*rms_1.*|.*rms_2.*|.*ln_f.*"
name = "transformer.h.19.attn.adapter_wte.weight"
# name = "transformer.h.0.rms_1.scale"
# name = "transformer.h.0.attn.c_attn.bias"
print(bool(re.match(pattern=pattern, string=name)))
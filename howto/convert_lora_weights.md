# Merging LoRA weights into base model weights

Purpose: By merging our selected LoRA weights into the base model weights, we can benefit from all base model optimisation such as quantisation (available in this repo), pruning, caching, etc.


## How to run?

After you have finish finetuning using LoRA, select your weight and run the converter script:

```bash
python scripts/convert_lora_weights.py --lora_path out/lora/your-folder/your-weight-name.pth
```

The converted base weight file will be saved into the same folder with the name `{your-weight-name}-lora-merged-weights.pth`. Now you can run `generate.py` with the merged weights and apply quantisation:

```bash
python generate.py --checkpoint_path out/lora/your-folder/your-weight-name-lora-merged-weights.pth --quantize llm.int8
```


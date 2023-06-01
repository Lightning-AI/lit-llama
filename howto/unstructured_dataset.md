# Finetuning on an unstructured dataset

While most scripts were made to finetune on instruction datasets, it is possible to finetune on any dataset. This is useful for experimentation while not being as expensive as training a full model.

This guide is only to prepare the finetuning, as either LoRA or Adapter-v1 methods support this dataset type!

## Preparation

1. Gather your text into an input file named `input.txt`
2. Divide the data into training and validation sets using the following script:

    ```bash
    python scripts/prepare_any_text.py
    ```

3. Modify relevant scripts for your finetuning method under `finetune/` and `evaluate/`, setting the `instruction_tuning` variable to `False`

And then you're set! Proceed to run the [LoRA guide](./finetune_lora.md) or [Adapter v1 guide](./finetune_adapter.md).
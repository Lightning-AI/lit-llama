# Generation with LoRA finetuned model

## Single generation

For a single generation (i.e. multiple inferences until a full answer has been generated)
with our lora finetuned model, use `lora.py`.

The lora-weights can be set with the `lora_path` argument, which defaults to booking_negotiation
("/workspace/lit-llama/out/lora/mediform_bookingnegotiation/lit-llama-lora-finetuned.pth").

To use the same settings as our evaluation (see below), use `top_k 1, temperature 0.1`.

Adding our full prompt through CLI is a bit difficult, so you can create your prompt as text file
and pass it to the python call via `$(cat text_file)`.

`python lora.py --prompt "$(cat prompt.txt)" --top_k 1 --temperature 0.1`


## Batch predictions

To perform generations for multiple instances with our lora finetuned model, use `lora_predictions.py`.

The `destination_path` selects the lit-llama data path to find the test set (`data_file_name`)
and to write the evaluation results (`evaluation_<data_file_name>`), and defaults to "/workspace/lit-llama/data/mediform_bookingnegotiation".
`data_file_name` can either be a json file in the format of the train set (for instance the default 
`230531_dialogs_mini.json`), or a pre-processed (see `prepare_*` scripts in folder `scripts`)
pytorch file (for instance `test.pt`) to perform evaluation on the same examples that have been
used for the test set during finetuning. 

The result is stored in `prediction_<data_file_name>`.


## Evaluation

To perform a domain specific evaluation of predictions stored in `prediction_data_file_name`, use `evaluation.py`.

Evaluation is independent of the model that was used for predictions, so this script is not specific to lora.

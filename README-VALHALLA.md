## Environment

`source .venv/bin/activate` 

## Set up the weights

The official weights can be found in Valhalla: `/fs/sigyn0/bhaddow/models/llama`.

The modified weights will be saved on the local repo checkpoints folder (you can change the path, should we keep them in fs too (?)) which is the default path in the generate scripts.

`python scripts/convert_checkpoint.py --model_size 7B --ckpt_dir /fs/sigyn0/bhaddow/models/llama --output_dir checkpoints/lit-llama`

Once your run that you should have the following folder structure:

```
lit-llama/checkpoints/lit-llama/
    tokenizer.model
    7B/lit-llama.pth
```

TODO: Which I think you have to manually put the tokenizer.model into the lit-llama file (it saves it on checkpoints by default)

## Run inference with the `generate.py`

```bash
CUDA_VISIBLE_DEVICES=1 python generate.py --prompt "Hello, my name is"
```

So according to the README this loads it in full precision, unless it detects your GPU will choke, that then runs it on bfloat16 and takes 14GB, which checks out (13577MiB).

## Run sanity check translations

Prompts are in: `/mnt/startiger0/nbogoych/allprompt.tar.gz`

TODO###
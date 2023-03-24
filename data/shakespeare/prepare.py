import os
import requests
import numpy as np
from models.llama.tokenizer import Tokenizer


# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
if not os.path.exists(input_file_path):
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with open(input_file_path, "w") as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, "r") as f:
    data = f.read()
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

tokenizer = Tokenizer("/srv/data/checkpoints/llama/converted_meta/tokenizer.model")
train_ids = tokenizer.encode(train_data, bos=True, eos=False)
val_ids = tokenizer.encode(val_data, bos=True, eos=False)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens

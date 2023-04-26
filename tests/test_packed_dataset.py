import pytest
import os
import requests


def train_tokenizer(destination_path):
    destination_path.mkdir(parents=True, exist_ok=True)

    # download the tiny shakespeare dataset
    input_file_path = destination_path / "input.txt"
    if not input_file_path.exists():
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)

    from lit_llama import Tokenizer
    Tokenizer.train(
        input=input_file_path,
        destination=destination_path,
        vocab_size=100,
    )

    return destination_path / "tokenizer.model"


def test_packed_dataset(tmp_path):
    tokenizer_path = train_tokenizer(tmp_path)

    from lit_llama import Tokenizer
    tokenizer = Tokenizer(tokenizer_path)

    texts = [
      "The moment of truth is upon us.",
      "Time to open the fridge."
    ]

    from lit_llama.packed_dataset import PackedDatasetBuilder, PackedDataset, HDR_SIZE

    block_size = 10
    n_blocks = 2

    builder = PackedDatasetBuilder(
        outdir=tmp_path,
        prefix="packed_dataset",
        block_size=block_size,
        n_blocks=n_blocks,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=100,
    )

    text_ids = []

    for text in texts:
        text_ids = tokenizer.encode(text)
        assert text_ids[0] == tokenizer.bos_id
        builder.add_array(text_ids)

    filenames = builder.filenames

    assert len(filenames) == 2
    assert os.path.basename(filenames[0]) == "packed_dataset_0000000000.bin"
    assert os.path.basename(filenames[1]) == "packed_dataset_0000000001.bin"

    import numpy as np

    expected = [
        "The moment of truth is up",
        "on us. Time to open the fri"
    ]

    for filename, el in zip(filenames, expected):
        mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
        count = len(mmap) // np.dtype(builder.dtype).itemsize
        arr = np.frombuffer(
            mmap, dtype=builder.dtype, count=count, offset=0
        )
        where_bos = np.where(arr == tokenizer.bos_id)
        # we expect two BOS tokens, one per file
        assert len(where_bos) == 1
        assert tokenizer.decode(arr) == el

    dataset = PackedDataset(filenames=filenames, n_chunks=2, shuffle=False)

    expected = [
        "The moment of ",
        "truth is up",
        "on us. Tim",
        "e to open the fri",
    ]

    for item, el in zip(dataset, expected):
        assert tokenizer.decode(item) == el

    dataset = PackedDataset(filenames=filenames, n_chunks=2, seed=12345)

    for i, item in enumerate(dataset):
        block_idxs = iter(dataset)._block_idxs
        assert tokenizer.decode(item) == expected[block_idxs[i]]

    dataset = PackedDataset(filenames=filenames, n_chunks=1, seed=12345)

    for i, item in enumerate(dataset):
        block_idxs = iter(dataset)._block_idxs
        chunk_idx = i // n_blocks * n_blocks
        assert tokenizer.decode(item) == expected[chunk_idx + block_idxs[i % n_blocks]]

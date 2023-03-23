import argparse
import os
from transformers.models.llama.convert_llama_weights_to_hf import write_model,write_tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "13B", "30B", "65B", "tokenizer_only"],
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    args = parser.parse_args()
    if args.model_size != "tokenizer_only":
        write_model(
            model_path=os.path.join(args.output_dir, "llama-{}".format(args.model_size).lower()),
            input_base_path=os.path.join(args.input_dir, args.model_size),
            model_size=args.model_size,
        )
    write_tokenizer(
        tokenizer_path=os.path.join(args.output_dir, "llama-{}".format(args.model_size).lower()),
        input_tokenizer_path=os.path.join(args.input_dir, "tokenizer.model"),
    )


if __name__ == "__main__":
    main()

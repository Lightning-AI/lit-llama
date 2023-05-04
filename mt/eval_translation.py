"""
    This script has as input a csv file with the following columns:
    src
    ref
    mt
    Will compute some metrics and generate the columns:
    bleu
    chrf
    wmt22-comet-da

    And save the averages on a file that can be commited and save all teh results to an output csv.
"""
import json
import pandas as pd
from jsonargparse import CLI
from sacrebleu import sentence_bleu, sentence_chrf
from comet import download_model, load_from_checkpoint


def main_eval(
    input_csv_file: str = "./mt/data/wmt19_en-de_translated.csv",
    output_csv_file: str = "./mt/data/wmt19_en-de_eval.csv",
    results_json_file: str = "./mt/results/wmt19_en-de_results.json",
):
    df = pd.read_csv(input_csv_file)
    df['bleu'] = [
        sentence_bleu(hyp, [ref]).score
        for hyp, ref in zip(df['mt'], df['ref'])
    ]
    df['chrf'] = [
        sentence_chrf(hyp, [ref]).score
        for hyp, ref in zip(df['mt'], df['ref'])
    ]
    # comet
    data = [
        {
            "src": src,
            "ref": ref,
            "mt": mt,
        }
        for src, ref, mt in zip(df['src'], df['ref'], df['mt'])
    ]
    # taken from https://huggingface.co/Unbabel/wmt22-comet-da
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    model_output = model.predict(data, batch_size=8, gpus=1)
    df['wmt22-comet-da'] = model_output.scores
    # save df to output path
    df.to_csv(output_csv_file, index=False)
    # compact results
    results = {
        "translated": input_csv_file,
        "eval": output_csv_file,
        "samples": len(df),
        "bleu": df['bleu'].mean(),
        "chrf": df['chrf'].mean(),
        "wmt22-comet-da": df['wmt22-comet-da'].mean(),
    }
    with open(results_json_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    CLI(main_eval)

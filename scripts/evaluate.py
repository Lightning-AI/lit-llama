"""Basic script to evaluate metrics on a test set. Adapt this to your needs!"""
import torch
import os

from generate import generate
from lit_llama.adapter import LLaMA, LLaMAConfig, mark_only_adapter_as_trainable, adapter_state_dict
from lit_llama.tokenizer import Tokenizer
from scripts.prepare_alpaca import generate_prompt
from torchmetrics.text.bert import BERTScore
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    # fabric.print("Validating ...")
    # model.eval()
    # losses = torch.zeros(eval_iters)
    # for k in range(eval_iters):
    #     input_ids, targets = get_batch(fabric, val_data)
    #     logits = model(input_ids)
        

    
    preds = ["hello there", "general kenobi"]
    target = ["hello there", "master kenobi"]
    bertscore = BERTScore("roberta-large")
    score = bertscore(preds, target)
    from pprint import pprint
    rounded_score = {k: [round(v, 3) for v in vv] for k, vv in score.items()}
    pprint(rounded_score)



def generate_response(model, instruction):
    tokenizer = Tokenizer("checkpoints/lit-llama/tokenizer.model")
    sample = {"instruction": instruction, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=True)
    encoded = encoded[None, :]  # add batch dimension
    encoded = encoded.to(model.device)

    output = generate(
        model,
        idx=encoded,
        max_seq_length=block_size,
        max_new_tokens=100,
    )
    output = tokenizer.decode(output[0].cpu())
    return output.split("### Response:")[1].strip()


@torch.no_grad()
def validate(fabric, model, val_data) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    # produce an example:
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    
    output = generate_response(model, instruction)
    fabric.print(instruction)
    fabric.print(output)

    model.train()
    return out.item()




if __name__ == "__main__":
    main()

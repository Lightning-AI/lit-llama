import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import clip
import gradio as gr
import lightning as L
import torch
from PIL import Image

from finetune.adapter_v2_multi_modal import encode_image
from lit_llama.adapter_v2_multi_modal import LLaMA
from lit_llama.tokenizer import Tokenizer
from lit_llama.utils import lazy_load
from scripts.prepare_alpaca import generate_prompt

adapter_path = Path("out/adapter_v2/llava/iter-012799.pth")
pretrained_path = Path("checkpoints/lit-llama/7B/lit-llama.pth")
tokenizer_path = Path("checkpoints/lit-llama/tokenizer.model")

fabric = L.Fabric(devices=1, precision="bf16-true")

with fabric.init_module(empty_init=True):
    model = LLaMA.from_name("7B")
with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
    model.load_state_dict(pretrained_checkpoint, strict=False)
    model.load_state_dict(adapter_checkpoint, strict=False)

model.eval()
model = fabric.setup(model)
tokenizer = Tokenizer(tokenizer_path)

clip_model, clip_transform = clip.load("ViT-L/14")
clip_model = fabric.to_device(clip_model)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: int,
    temperature: float = 1.0,
    top_k = None,
    eos_id = None,
    **kwargs,
) -> torch.Tensor:

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = idx.size(0)
    T_new = T + max_new_tokens
    empty = torch.empty(T_new, dtype=idx.dtype, device=idx.device)
    empty[:T] = idx
    idx = empty

    # generate max_new_tokens tokens
    for t in range(T, T_new):
        # ignore the not-filled-yet tokens
        idx_cond = idx[:t]
        # if the sequence context is growing too long we must crop it at max_seq_length
        idx_cond = idx_cond if t <= max_seq_length else idx_cond[-max_seq_length:]

        # forward
        logits = model(idx_cond.view(1, -1), **kwargs)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[[-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # concatenate the new generation
        # https://github.com/pytorch/pytorch/issues/101936
        idx[t] = idx_next.item() if idx.device.type == "mps" else idx_next

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:t + 1]  # include the EOS token

    return idx


def predict(image, prompt, temperature):
    sample = {"instruction": prompt, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
    image = Image.open(image).convert("RGB")
    image = clip_transform(image)
    image = fabric.to_device(image).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        img_features = encode_image(clip_model, image)

    output = generate(
        model,
        encoded,
        max_new_tokens=256,
        max_seq_length=model.config.block_size,
        temperature=temperature,
        top_k=200,
        eos_id=tokenizer.eos_id,
        img_features=img_features,
    )

    output = tokenizer.decode(output)
    output = output.split("### Response:")[1].strip()
    return output


def create_layout():
    with gr.Blocks(theme="lightdefault") as main, gr.TabItem("LLaMA Image and Text"):
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Image", type="filepath")
                prompt = gr.Textbox(label="Prompt", lines=2)
                temperature = gr.Slider(label="Temperature", minimum=0, maximum=1, value=0.1)
                run = gr.Button("Run")
            with gr.Column():
                outputs = gr.Textbox(label="Response", lines=10)

        inputs = [image, prompt, temperature]
        examples = [
            ["COCO_test2014_000000251884.jpg", "Describe the image.", 0.1],
        ]

        gr.Examples(
            examples=examples,
            inputs=inputs,
            outputs=outputs,
            fn=predict,
        )
        run.click(fn=predict, inputs=inputs, outputs=outputs)
    return main


create_layout().queue().launch(share=True, server_port=8090)

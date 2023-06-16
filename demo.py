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

from finetune.adapter_v2_llava import clip_encode_image
from generate import generate
from lit_llama.adapter_v2 import LLaMA
from lit_llama.tokenizer import Tokenizer
from lit_llama.utils import lazy_load, llama_model_lookup
from scripts.prepare_alpaca import generate_prompt

adapter_path = Path("out/adapter_v2/llava/iter-012799.pth")
pretrained_path = Path("checkpoints/lit-llama/7B/lit-llama.pth")
tokenizer_path = Path("checkpoints/lit-llama/tokenizer.model")

fabric = L.Fabric(devices=1, precision="bf16-true")

with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
    name = llama_model_lookup(pretrained_checkpoint)

    with fabric.init_module(empty_init=True):
        model = LLaMA.from_name(name)

    # 1. Load the pretrained weights
    model.load_state_dict(pretrained_checkpoint, strict=False)
    # 2. Load the fine-tuned adapter weights
    model.load_state_dict(adapter_checkpoint, strict=False)

model.eval()
model = fabric.setup(model)
tokenizer = Tokenizer(tokenizer_path)

clip_model, clip_transform = clip.load("ViT-L/14")
clip_model = fabric.to_device(clip_model)


def predict(image, prompt, temperature):
    sample = {"instruction": prompt, "input": ""}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
    image = Image.open(image).convert("RGB")
    image = clip_transform(image)
    image = fabric.to_device(image).unsqueeze(0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        img_features = clip_encode_image(clip_model, image)

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
    with gr.Blocks() as main, gr.TabItem("LLaMA Image and Text"):
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

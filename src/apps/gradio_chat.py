import argparse

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model


tokenizer, model = None, None


def _format_prompt(instruction: str, context: str) -> str:
    if context.strip():
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{context}\n\n"
            "### Response:\n"
        )
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def chat_fn(instruction: str, context: str):
    if tokenizer is None or model is None:
        raise RuntimeError("Model not loaded. Launch the app with build_app().")
    prompt = _format_prompt(instruction, context)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    return generated


def build_app(model_dir="outputs/v2_qlora_middle"):
    global tokenizer, model
    tokenizer, model = _load(model_dir)

    with gr.Blocks() as demo:
        gr.Markdown("# Small-Decoder Chatbot (single-turn)")
        gr.Textbox(
            value=model_dir,
            label="Model directory (loaded at launch)",
            interactive=False,
        )
        instruction = gr.Textbox(lines=4, label="Instruction")
        context = gr.Textbox(lines=4, label="Optional Input")
        output = gr.Textbox(lines=8, label="Response")
        trigger = gr.Button("Generate")
        trigger.click(fn=chat_fn, inputs=[instruction, context], outputs=output)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", default="outputs/v2_qlora_middle")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo = build_app(args.model_dir)
    demo.launch(server_name="0.0.0.0", server_port=args.port)

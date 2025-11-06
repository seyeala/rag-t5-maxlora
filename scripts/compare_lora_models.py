"""Gradio app for comparing a base seq2seq model with a LoRA adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import gradio as gr
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from rag_t5.config import load_settings
from rag_t5.models.loader import load_base
from apps.gradio_chat import _format_prompt


def _load_models(adapter_dir: Path):
    cfg = load_settings("configs/defaults.toml")
    base_model, base_tokenizer, device = load_base(cfg.base_model)
    model_dtype = next(base_model.parameters()).dtype

    peft_config = PeftConfig.from_pretrained(adapter_dir)
    ft_tokenizer = AutoTokenizer.from_pretrained(
        peft_config.base_model_name_or_path,
        use_fast=True,
    )
    ft_base = AutoModelForSeq2SeqLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=model_dtype,
    )
    ft_model = PeftModel.from_pretrained(ft_base, adapter_dir).to(device)
    ft_model.eval()

    return (base_model, base_tokenizer), (ft_model, ft_tokenizer), device


def _decode(tokenizer, generated):
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


def compare_models(
    instruction: str,
    context: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    *,
    base_bundle: Tuple,
    ft_bundle: Tuple,
    device,
):
    if not instruction.strip():
        return "", ""

    prompt = _format_prompt(instruction, context)
    (base_model, base_tokenizer), (ft_model, ft_tokenizer) = base_bundle, ft_bundle

    base_inputs = base_tokenizer(prompt, return_tensors="pt").to(device)
    ft_inputs = ft_tokenizer(prompt, return_tensors="pt").to(device)

    generation_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=True,
    )

    with torch.inference_mode():
        base_output = base_model.generate(**base_inputs, **generation_kwargs)
        ft_output = ft_model.generate(**ft_inputs, **generation_kwargs)

    base_text = _decode(base_tokenizer, base_output[0])
    ft_text = _decode(ft_tokenizer, ft_output[0])

    return base_text, ft_text


def build_demo(adapter_dir: str):
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    base_bundle, ft_bundle, device = _load_models(adapter_path)

    with gr.Blocks() as demo:
        gr.Markdown("## Base vs fine-tuned comparison")
        with gr.Row():
            instruction = gr.Textbox(lines=4, label="Instruction")
            context = gr.Textbox(lines=4, label="Optional input")
        with gr.Row():
            max_new_tokens = gr.Slider(
                minimum=32,
                maximum=512,
                step=32,
                value=256,
                label="Max new tokens",
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                step=0.1,
                value=0.7,
                label="Temperature",
            )
            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                step=0.05,
                value=0.9,
                label="Top-p",
            )
        compare = gr.Button("Generate")
        with gr.Row():
            base_box = gr.Textbox(label="Base model", lines=8)
            ft_box = gr.Textbox(label="Fine-tuned", lines=8)
        compare.click(
            fn=lambda inst, ctx, max_tokens, temp, topp: compare_models(
                inst,
                ctx,
                max_tokens,
                temp,
                topp,
                base_bundle=base_bundle,
                ft_bundle=ft_bundle,
                device=device,
            ),
            inputs=[instruction, context, max_new_tokens, temperature, top_p],
            outputs=[base_box, ft_box],
        )

    return demo


__all__ = ["build_demo", "compare_models"]

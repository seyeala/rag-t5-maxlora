"""Gradio app for comparing a base seq2seq model with one or more LoRA adapters."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import gradio as gr
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from apps.gradio_chat import _format_prompt
from rag_t5.config import load_settings
from rag_t5.models.loader import load_base


ModelBundle = Tuple[str, torch.nn.Module, AutoTokenizer]


def _load_adapter(
    adapter_dir: Path,
    *,
    device: str,
    dtype: torch.dtype,
) -> ModelBundle:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_name = peft_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_name, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
    model.eval()

    label = f"LoRA: {adapter_dir.name} ({base_name})"
    return label, model, tokenizer


def _load_models(
    adapter_dirs: Sequence[Path],
    *,
    base_model_id: str | None = None,
) -> Tuple[List[ModelBundle], str]:
    cfg = load_settings("configs/defaults.toml")
    target_base = base_model_id or cfg.base_model
    base_model, base_tokenizer, device = load_base(target_base)
    base_model.eval()
    dtype = next(base_model.parameters()).dtype

    bundles: List[ModelBundle] = [(f"Base: {target_base}", base_model, base_tokenizer)]
    for adapter_dir in adapter_dirs:
        bundles.append(_load_adapter(adapter_dir, device=device, dtype=dtype))

    return bundles, device


def _generate_text(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    device: str,
    generation_kwargs: dict,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)
    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def compare_models(
    instruction: str,
    context: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    *,
    bundles: Sequence[ModelBundle],
    device: str,
) -> Tuple[str, ...]:
    if not instruction.strip():
        return tuple("" for _ in bundles)

    prompt = _format_prompt(instruction, context)
    generation_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=do_sample,
    )

    outputs = []
    for _, model, tokenizer in bundles:
        text = _generate_text(
            model,
            tokenizer,
            prompt,
            device=device,
            generation_kwargs=generation_kwargs,
        )
        outputs.append(text)
    return tuple(outputs)


def build_demo(
    adapter_dirs: Iterable[str],
    *,
    base_model_id: str | None = None,
) -> gr.Blocks:
    adapter_paths = [Path(path) for path in adapter_dirs]
    bundles, device = _load_models(adapter_paths, base_model_id=base_model_id)

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
                minimum=0.0,
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
            sampling = gr.Checkbox(label="Enable sampling", value=True)

        compare = gr.Button("Generate")

        with gr.Row():
            output_boxes = [
                gr.Textbox(label=label, lines=8)
                for label, _, _ in bundles
            ]

        compare.click(
            fn=lambda inst, ctx, max_tokens, temp, topp, sample: compare_models(
                inst,
                ctx,
                max_tokens,
                temp,
                topp,
                sample,
                bundles=bundles,
                device=device,
            ),
            inputs=[instruction, context, max_new_tokens, temperature, top_p, sampling],
            outputs=output_boxes,
        )

    return demo


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "adapter",
        nargs="+",
        help="One or more directories containing LoRA adapters to compare.",
    )
    parser.add_argument(
        "--base-model",
        dest="base_model",
        help="Optional base model ID to use for the baseline column.",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Share the Gradio app.")
    return parser.parse_args()


def main():
    args = _parse_args()
    demo = build_demo(args.adapter, base_model_id=args.base_model)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


__all__ = ["build_demo", "compare_models", "main"]


if __name__ == "__main__":
    main()

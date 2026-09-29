from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


def _model_class(config):
    return AutoModelForSeq2SeqLM if getattr(config, "is_encoder_decoder", False) else AutoModelForCausalLM


def load_inference_model(model_path: str):
    path = Path(model_path)
    is_adapter = path.is_dir() and (path / "adapter_config.json").is_file()

    if is_adapter:
        peft_config = PeftConfig.from_pretrained(model_path)
        base_id = peft_config.base_model_name_or_path
        config = AutoConfig.from_pretrained(base_id)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        base = _model_class(config).from_pretrained(
            base_id,
            config=config,
            dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = _model_class(config).from_pretrained(
            model_path,
            config=config,
            dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        )

    model.eval()
    return tokenizer, model

from typing import Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_base(
    model_name: str = "google/flan-t5-small",
    device: str | None = None,
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer, str]:
    """Load a base seq2seq model and tokenizer on the target device."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device

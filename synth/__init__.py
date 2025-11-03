"""Synthetic data generation utilities."""

from .gen_qa_offline import generate_sft_examples, preview_examples, save_sft_dataset

__all__ = ["generate_sft_examples", "save_sft_dataset", "preview_examples"]

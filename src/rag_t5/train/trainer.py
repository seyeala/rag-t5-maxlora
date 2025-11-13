"""High level training entrypoint used by notebooks and scripts.

This module provides a thin wrapper around the utilities implemented in
:mod:`train.common`.  The goal is to expose a stable ``rag_t5.train`` import
path so external users can simply run ``from rag_t5.train.trainer import
train`` without worrying about the internal project layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch

from rag_t5.config import AppCfg

from train.common import (
    LoRA_TARGETS_ATT_MLP,
    TrainConfig as _TrainerConfig,
    apply_lora_everywhere,
    count_trainable_params,
    freeze_lora_outside,
    get_num_layers_and_attr,
    last_n_indices,
    load_fp_model,
    load_tokenizer,
    make_dataset,
    run_trainer,
    resolve_bf16,
    unfreeze_lm_head,
)


@dataclass
class TrainConfig:
    """Configuration for the :func:`train` helper.

    The fields mirror the options used in the reference training scripts while
    keeping reasonable defaults so the helper can be invoked with a minimal
    amount of boilerplate.
    """

    model_id: str
    train_path: str | None
    valid_path: str | None
    out_dir: str
    max_length: int = 512
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    label_smoothing_factor: float = 0.0
    gradient_checkpointing: bool = False
    optim: str | None = None
    lr_scheduler_type: str | None = None
    bf16: bool | None = None
    logging_steps: int = 10
    save_strategy: str = "no"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: Sequence[str] = field(
        default_factory=lambda: tuple(LoRA_TARGETS_ATT_MLP)
    )
    last_n_lora_layers: int | None = 2

    @classmethod
    def from_app_config(
        cls,
        settings: AppCfg,
        *,
        train_path: str,
        valid_path: str | None,
        out_dir: str,
        **overrides,
    ) -> "TrainConfig":
        """Build a :class:`TrainConfig` from :func:`rag_t5.config.load_settings` output.

        Parameters
        ----------
        settings:
            Parsed application configuration returned by
            :func:`rag_t5.config.load_settings`.
        train_path, valid_path, out_dir:
            Dataset paths and output directory passed through to the trainer.
        overrides:
            Additional keyword arguments that override values resolved from the
            config (e.g. ``num_train_epochs=5``).
        """

        train_cfg = settings.train
        lora_cfg = settings.lora

        resolved = dict(
            model_id=train_cfg.model_id or settings.base_model,
            train_path=train_path,
            valid_path=valid_path,
            out_dir=out_dir,
            max_length=train_cfg.src_max_len,
            num_train_epochs=float(train_cfg.epochs),
            per_device_train_batch_size=train_cfg.batch_size,
            gradient_accumulation_steps=train_cfg.grad_accum,
            learning_rate=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            warmup_ratio=train_cfg.warmup_ratio,
            label_smoothing_factor=train_cfg.label_smoothing,
            gradient_checkpointing=train_cfg.gradient_checkpointing,
            lora_r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            lora_targets=tuple(lora_cfg.target_modules),
        )

        resolved.update(overrides)

        return cls(**resolved)


def train(config: TrainConfig):
    """Run a LoRA fine-tuning session using :class:`~transformers.Trainer`.

    Parameters
    ----------
    config:
        Training configuration describing the dataset, model and optimisation
        hyper-parameters.

    Returns
    -------
    model, tokenizer, dict
        The fine-tuned model, its tokenizer and a dictionary of efficiency
        statistics produced by :func:`train.common.run_trainer`.
    """

    use_bf16 = resolve_bf16(config.bf16)
    tokenizer = load_tokenizer(config.model_id)
    model_dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = load_fp_model(config.model_id, dtype=model_dtype)

    if config.train_path is None and config.valid_path is None:
        Path(config.out_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(config.out_dir)
        tokenizer.save_pretrained(config.out_dir)
        trainable, pct = count_trainable_params(model)
        efficiency = {
            "trainable_params": trainable,
            "trainable_pct": pct,
            "peak_vram_gb": 0.0,
            "wall_time_s": 0.0,
        }
        return model, tokenizer, efficiency

    model = apply_lora_everywhere(
        model,
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        targets=list(config.lora_targets),
    )

    if config.last_n_lora_layers is not None:
        num_layers, _ = get_num_layers_and_attr(model)
        allowed = set(last_n_indices(num_layers, config.last_n_lora_layers))
        freeze_lora_outside(model, allowed)

    unfreeze_lm_head(model)

    trainer_config = _TrainerConfig(
        model_id=config.model_id,
        train_path=config.train_path,
        valid_path=config.valid_path,
        out_dir=config.out_dir,
        max_length=config.max_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        label_smoothing_factor=config.label_smoothing_factor,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=use_bf16,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
    )

    train_dataset, valid_dataset = make_dataset(
        tokenizer, config.train_path, config.valid_path, config.max_length
    )

    efficiency = run_trainer(
        model, tokenizer, train_dataset, valid_dataset, trainer_config
    )

    return model, tokenizer, efficiency


__all__ = ["TrainConfig", "train"]

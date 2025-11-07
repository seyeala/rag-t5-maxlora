"""Branch-based fine-tuning workflows for the three-variant comparison.

This module implements a high-level orchestration layer that mirrors the
assignment described in :mod:`doc/main.tex`.  It prepares three branches that
share the same dataset and optimisation budget while differing in which
parameters are trainable:

``Branch 1`` – Middle-block full-precision fine-tuning.
``Branch 2`` – QLoRA adapters on the middle third of layers.
``Branch 3`` – Head-only baseline (optionally extended with shallow LoRA).

Each branch records base (pre-fine-tuning) metrics, fine-tuned metrics and the
efficiency statistics written by the lower-level trainer utilities.
"""

from __future__ import annotations

import json
import numbers
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping

import torch
from peft import prepare_model_for_kbit_training

from train.common import (
    LoRA_TARGETS_ATT_MLP,
    TrainConfig as _TrainerConfig,
    apply_lora_everywhere,
    build_trainer,
    count_trainable_params,
    freeze_lora_outside,
    get_num_layers_and_attr,
    load_4bit_model,
    load_fp_model,
    load_tokenizer,
    make_dataset,
    middle_third_indices,
    peak_vram_gb,
    reset_vram,
    resolve_bf16,
    unfreeze_lm_head,
)


class BranchStrategy(Enum):
    """Enumeration of the three comparison branches."""

    BRANCH1_MIDDLE_BLOCK = "branch1"
    BRANCH2_QLORA = "branch2"
    BRANCH3_BASELINE = "branch3"

    @property
    def display_name(self) -> str:
        return {
            BranchStrategy.BRANCH1_MIDDLE_BLOCK: "Branch 1 · Middle-Block FT",
            BranchStrategy.BRANCH2_QLORA: "Branch 2 · Middle QLoRA",
            BranchStrategy.BRANCH3_BASELINE: "Branch 3 · Head Baseline",
        }[self]

    @property
    def description(self) -> str:
        return {
            BranchStrategy.BRANCH1_MIDDLE_BLOCK:
                "Unfreeze only the middle third of transformer blocks plus lm_head.",
            BranchStrategy.BRANCH2_QLORA:
                "Attach QLoRA adapters to the middle third of layers (lm_head trainable).",
            BranchStrategy.BRANCH3_BASELINE:
                "Freeze the backbone and optimise lm_head (optional shallow LoRA).",
        }[self]


@dataclass
class BranchComparisonConfig:
    """Configuration shared across the three-branch comparison."""

    model_id: str
    train_path: str
    valid_path: str
    out_dir: str
    max_length: int = 512
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    bf16: bool | None = None
    logging_steps: int = 10
    save_strategy: str = "no"
    max_steps: int | None = None
    train_limit: int | None = None
    valid_limit: int | None = None
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: Iterable[str] = field(
        default_factory=lambda: tuple(LoRA_TARGETS_ATT_MLP)
    )
    branch_overrides: Dict[BranchStrategy, Dict[str, object]] = field(
        default_factory=dict
    )
    eval_before_train: bool = True
    eval_after_train: bool = True


@dataclass
class BranchResult:
    """Outcome of a single branch run."""

    strategy: BranchStrategy
    out_dir: str
    efficiency: Dict[str, float]
    base_metrics: Dict[str, float | int] | None
    final_metrics: Dict[str, float | int] | None
    training_metrics: Dict[str, float | int] | None
    notes: Dict[str, object] = field(default_factory=dict)

    @property
    def display_name(self) -> str:  # pragma: no cover - simple passthrough
        return self.strategy.display_name


def _clean_metrics(metrics: Mapping[str, object] | None) -> Dict[str, float | int]:
    cleaned: Dict[str, float | int] = {}
    if not metrics:
        return cleaned
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            cleaned[key] = value.item()
        elif isinstance(value, numbers.Integral):
            cleaned[key] = int(value)
        elif isinstance(value, numbers.Real):
            cleaned[key] = float(value)
        else:
            cleaned[key] = value  # May be JSON-serialisable already.
    return cleaned


def _resolve_attr_path(model, path: str):
    obj = model
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def _select_layer_indices(
    num_layers: int, branch_params: MutableMapping[str, object]
) -> list[int]:
    custom = branch_params.pop("layer_indices", None)
    if custom is None:
        return list(middle_third_indices(num_layers))
    indices: set[int] = set()
    for raw in custom:
        try:
            idx = int(raw)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"layer_indices must be integers, received {raw!r}") from exc
        if 0 <= idx < num_layers:
            indices.add(idx)
    return sorted(indices)


def _prepare_middle_block_branch(
    cfg: _TrainerConfig,
    branch_params: MutableMapping[str, object],
) -> tuple[object, Dict[str, object]]:
    use_bf16 = resolve_bf16(cfg.bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = load_fp_model(cfg.model_id, dtype=dtype)
    for param in model.parameters():
        param.requires_grad = False
    num_layers, attr = get_num_layers_and_attr(model)
    layers = _resolve_attr_path(model, attr)
    indices = _select_layer_indices(num_layers, branch_params)
    for idx in indices:
        for param in layers[idx].parameters():
            param.requires_grad = True
    unfreeze_lm_head(model)
    notes = {
        "layer_attr": attr,
        "trainable_layers": indices,
        "dtype": str(dtype),
    }
    return model, notes


def _prepare_qlora_branch(
    cfg: _TrainerConfig,
    comparison_cfg: BranchComparisonConfig,
    branch_params: MutableMapping[str, object],
) -> tuple[object, Dict[str, object]]:
    model = load_4bit_model(cfg.model_id)
    model = prepare_model_for_kbit_training(model)

    r = int(branch_params.pop("lora_r", comparison_cfg.lora_r))
    alpha = int(branch_params.pop("lora_alpha", comparison_cfg.lora_alpha))
    dropout = float(branch_params.pop("lora_dropout", comparison_cfg.lora_dropout))
    targets = branch_params.pop("lora_targets", tuple(comparison_cfg.lora_targets))

    model = apply_lora_everywhere(
        model,
        r=r,
        alpha=alpha,
        dropout=dropout,
        targets=list(targets),
    )

    num_layers, attr = get_num_layers_and_attr(model)
    indices = _select_layer_indices(num_layers, branch_params)
    freeze_lora_outside(model, set(indices))
    unfreeze_lm_head(model)
    notes = {
        "layer_attr": attr,
        "trainable_layers": indices,
        "lora_r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "lora_targets": list(targets),
    }
    return model, notes


def _prepare_baseline_branch(
    cfg: _TrainerConfig,
    comparison_cfg: BranchComparisonConfig,
    branch_params: MutableMapping[str, object],
) -> tuple[object, Dict[str, object]]:
    top_lora_layers = branch_params.pop("top_lora_layers", 0)
    if top_lora_layers:
        model = load_4bit_model(cfg.model_id)
        model = prepare_model_for_kbit_training(model)
        r = int(branch_params.pop("lora_r", comparison_cfg.lora_r))
        alpha = int(branch_params.pop("lora_alpha", comparison_cfg.lora_alpha))
        dropout = float(branch_params.pop("lora_dropout", comparison_cfg.lora_dropout))
        targets = branch_params.pop("lora_targets", tuple(comparison_cfg.lora_targets))
        model = apply_lora_everywhere(
            model,
            r=r,
            alpha=alpha,
            dropout=dropout,
            targets=list(targets),
        )
        num_layers, attr = get_num_layers_and_attr(model)
        if "layer_indices" in branch_params:
            allowed = _select_layer_indices(num_layers, branch_params)
        else:
            allowed = list(range(max(0, num_layers - top_lora_layers), num_layers))
        freeze_lora_outside(model, set(allowed))
        unfreeze_lm_head(model)
        notes = {
            "layer_attr": attr,
            "trainable_layers": allowed,
            "lora_r": r,
            "lora_alpha": alpha,
            "lora_dropout": dropout,
            "lora_targets": list(targets),
            "top_lora_layers": top_lora_layers,
        }
        return model, notes

    use_bf16 = resolve_bf16(cfg.bf16)
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = load_fp_model(cfg.model_id, dtype=dtype)
    for param in model.parameters():
        param.requires_grad = False
    unfreeze_lm_head(model)
    notes = {
        "layer_attr": None,
        "trainable_layers": [],
        "dtype": str(dtype),
    }
    return model, notes


def _make_branch_train_config(
    comparison_cfg: BranchComparisonConfig,
    strategy: BranchStrategy,
) -> tuple[_TrainerConfig, Dict[str, object]]:
    overrides = dict(comparison_cfg.branch_overrides.get(strategy, {}))
    cfg_kwargs = {
        "model_id": comparison_cfg.model_id,
        "train_path": comparison_cfg.train_path,
        "valid_path": comparison_cfg.valid_path,
        "out_dir": str(Path(comparison_cfg.out_dir) / strategy.value),
        "max_length": comparison_cfg.max_length,
        "per_device_train_batch_size": comparison_cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": comparison_cfg.gradient_accumulation_steps,
        "num_train_epochs": comparison_cfg.num_train_epochs,
        "learning_rate": comparison_cfg.learning_rate,
        "bf16": comparison_cfg.bf16,
        "logging_steps": comparison_cfg.logging_steps,
        "save_strategy": comparison_cfg.save_strategy,
        "max_steps": comparison_cfg.max_steps,
        "train_limit": comparison_cfg.train_limit,
        "valid_limit": comparison_cfg.valid_limit,
    }
    recognised = set(cfg_kwargs.keys())
    branch_params: Dict[str, object] = {}
    for key, value in list(overrides.items()):
        if key in recognised:
            cfg_kwargs[key] = value
            overrides.pop(key)
    for key in ("lora_r", "lora_alpha", "lora_dropout", "lora_targets", "layer_indices", "top_lora_layers"):
        if key in overrides:
            branch_params[key] = overrides.pop(key)
    if overrides:
        raise ValueError(
            f"Unsupported override keys for {strategy.display_name}: {sorted(overrides.keys())}"
        )
    trainer_cfg = _TrainerConfig(**cfg_kwargs)
    return trainer_cfg, branch_params


def _prepare_branch(
    strategy: BranchStrategy,
    trainer_cfg: _TrainerConfig,
    comparison_cfg: BranchComparisonConfig,
    branch_params: MutableMapping[str, object],
) -> tuple[object, Dict[str, object]]:
    if strategy is BranchStrategy.BRANCH1_MIDDLE_BLOCK:
        return _prepare_middle_block_branch(trainer_cfg, branch_params)
    if strategy is BranchStrategy.BRANCH2_QLORA:
        return _prepare_qlora_branch(trainer_cfg, comparison_cfg, branch_params)
    if strategy is BranchStrategy.BRANCH3_BASELINE:
        return _prepare_baseline_branch(trainer_cfg, comparison_cfg, branch_params)
    raise ValueError(f"Unhandled strategy: {strategy}")


def run_three_branch_comparison(
    config: BranchComparisonConfig,
) -> Dict[BranchStrategy, BranchResult]:
    """Execute the three-branch fine-tuning comparison.

    The function returns a dictionary mapping each :class:`BranchStrategy` to the
    recorded results.  Each branch creates its own output directory under
    ``config.out_dir`` and writes ``efficiency.json`` (for compatibility with
    existing tooling) plus ``comparison.json`` with evaluation snapshots.
    """

    results: Dict[BranchStrategy, BranchResult] = {}

    for strategy in BranchStrategy:
        trainer_cfg, branch_params = _make_branch_train_config(config, strategy)
        tokenizer = load_tokenizer(trainer_cfg.model_id)
        train_ds, valid_ds = make_dataset(
            tokenizer,
            trainer_cfg.train_path,
            trainer_cfg.valid_path,
            trainer_cfg.max_length,
            train_limit=trainer_cfg.train_limit,
            valid_limit=trainer_cfg.valid_limit,
        )

        model, notes = _prepare_branch(strategy, trainer_cfg, config, branch_params)
        trainer, _ = build_trainer(model, tokenizer, train_ds, valid_ds, trainer_cfg)

        base_metrics = (
            _clean_metrics(trainer.evaluate()) if config.eval_before_train else None
        )

        reset_vram()
        start = time.time()
        train_output = trainer.train()
        wall = time.time() - start
        vram = peak_vram_gb()
        trainable, pct = count_trainable_params(model)

        Path(trainer_cfg.out_dir).mkdir(parents=True, exist_ok=True)
        efficiency = {
            "trainable_params": trainable,
            "trainable_pct": pct,
            "peak_vram_gb": vram,
            "wall_time_s": wall,
        }

        with open(
            Path(trainer_cfg.out_dir) / "efficiency.json", "w", encoding="utf-8"
        ) as fp:
            json.dump(efficiency, fp, indent=2)

        final_metrics = (
            _clean_metrics(trainer.evaluate()) if config.eval_after_train else None
        )
        training_metrics = _clean_metrics(getattr(train_output, "metrics", {}))

        comparison_payload = {
            "strategy": strategy.display_name,
            "description": strategy.description,
            "base_metrics": base_metrics,
            "final_metrics": final_metrics,
            "training_metrics": training_metrics,
            "efficiency": efficiency,
            "notes": notes,
        }
        with open(
            Path(trainer_cfg.out_dir) / "comparison.json", "w", encoding="utf-8"
        ) as fp:
            json.dump(comparison_payload, fp, indent=2)

        model.save_pretrained(trainer_cfg.out_dir)
        tokenizer.save_pretrained(trainer_cfg.out_dir)

        results[strategy] = BranchResult(
            strategy=strategy,
            out_dir=trainer_cfg.out_dir,
            efficiency=efficiency,
            base_metrics=base_metrics,
            final_metrics=final_metrics,
            training_metrics=training_metrics or None,
            notes=notes,
        )

        # Release GPU memory before proceeding to the next branch.
        reset_vram()

    return results


__all__ = [
    "BranchStrategy",
    "BranchComparisonConfig",
    "BranchResult",
    "run_three_branch_comparison",
]


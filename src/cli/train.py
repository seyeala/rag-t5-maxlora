"""Typer CLI entrypoint for fine-tuning via the rag_t5.train helpers."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from rag_t5.config import load_settings
from rag_t5.utils.seed import set_seed

app = typer.Typer(help="Fine-tune a base model using the training section of the config file.")

_DEFAULT_TRAIN = "data/processed/alpaca_train.jsonl"
_DEFAULT_VALID = "data/processed/alpaca_valid.jsonl"
_DEFAULT_OUT = "artifacts/models/sft"


def _load_trainer_components():
    from rag_t5.train import TrainConfig as TrainerConfig
    from rag_t5.train import train as run_train

    return TrainerConfig, run_train


def _resolve_path(path: str, *, param_hint: str) -> str:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise typer.BadParameter(f"File not found: {resolved}", param_hint=param_hint)
    return str(resolved)


def _resolve_optional_path(path: str | None, *, param_hint: str) -> str | None:
    if path is None:
        return None
    return _resolve_path(path, param_hint=param_hint)


@app.command()
def sft(
    train_path: str = typer.Option(_DEFAULT_TRAIN, "--train-path", help="Path to the training JSONL dataset."),
    valid_path: str | None = typer.Option(
        _DEFAULT_VALID,
        "--valid-path",
        help="Optional validation JSONL dataset. Pass an empty string to skip validation.",
    ),
    out_dir: str = typer.Option(
        _DEFAULT_OUT,
        "--out-dir",
        help="Directory where the fine-tuned adapter and tokenizer will be written.",
    ),
    config: str = typer.Option("configs/defaults.toml", "--config", "-c", help="Base configuration TOML."),
    override: str | None = typer.Option(None, "--override", "-O", help="Optional override TOML."),
    epochs: float | None = typer.Option(None, "--epochs", help="Override the configured number of epochs."),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        help="Override the per-device batch size specified in the config.",
    ),
    grad_accum: int | None = typer.Option(
        None,
        "--grad-accum",
        help="Override gradient accumulation steps from the config.",
    ),
    lr: float | None = typer.Option(None, "--lr", help="Override the learning rate."),
    max_length: int | None = typer.Option(None, "--max-length", help="Override the maximum source length."),
    seed: int = typer.Option(13, "--seed", help="Random seed passed to the training utilities."),
    logging_steps: int | None = typer.Option(
        None,
        "--logging-steps",
        help="Override how frequently Trainer logs progress.",
    ),
) -> None:
    """Fine-tune the base model using Alpaca-style JSONL data."""

    cfg = load_settings(config, override)
    train_cfg = cfg.train
    if train_cfg.task.lower() != "sft":
        raise typer.BadParameter(
            f"Only the 'sft' task is supported, received {train_cfg.task!r}.", param_hint="[train].task"
        )

    resolved_train = _resolve_path(train_path, param_hint="--train-path")
    resolved_valid = _resolve_optional_path(valid_path or None, param_hint="--valid-path")

    set_seed(seed)

    trainer_cls, run_train = _load_trainer_components()

    trainer_cfg = trainer_cls(
        model_id=train_cfg.model_id or cfg.base_model,
        train_path=resolved_train,
        valid_path=resolved_valid,
        out_dir=out_dir,
        max_length=max_length or train_cfg.src_max_len,
        num_train_epochs=epochs or train_cfg.epochs,
        per_device_train_batch_size=batch_size or train_cfg.batch_size,
        gradient_accumulation_steps=grad_accum or train_cfg.grad_accum,
        learning_rate=lr or train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        warmup_ratio=train_cfg.warmup_ratio,
        label_smoothing_factor=train_cfg.label_smoothing,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        logging_steps=logging_steps or trainer_cls.logging_steps,
    )

    _, _, efficiency = run_train(trainer_cfg)

    typer.echo(f"Training complete → {out_dir}")
    typer.echo(json.dumps(efficiency, indent=2))


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    app()

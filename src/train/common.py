from __future__ import annotations

import inspect
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

try:
    from transformers.utils import (
        is_torch_bf16_cpu_available,
        is_torch_bf16_gpu_available,
    )
except (ImportError, AttributeError):  # pragma: no cover - fallback for older versions
    def is_torch_bf16_gpu_available():
        if not torch.cuda.is_available():
            return False
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(checker):
            try:
                return bool(checker())
            except TypeError:
                return False
        return False

    def is_torch_bf16_cpu_available():
        return False


logger = logging.getLogger(__name__)


# -------------------- Data IO --------------------

def stream_jsonl(path: str):
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                yield json.loads(line)


def load_jsonl_pair(path: str):
    return [
        {"prompt": record["prompt"], "answer": record["answer"]}
        for record in stream_jsonl(path)
    ]


def build_example(record):
    text = record["prompt"] + record["answer"]
    return text, len(record["prompt"])


class PromptMaskedDataset(Dataset):
    def __init__(self, records, tokenizer, max_len=512, *, is_encoder_decoder=False):
        self.data = self._tokenize(records, tokenizer, max_len, is_encoder_decoder)

    @staticmethod
    def _tokenize(records, tokenizer, max_len, is_encoder_decoder):
        input_ids, attention_mask, labels = [], [], []
        for record in records:
            if is_encoder_decoder:
                prompt_enc = tokenizer(
                    record["prompt"],
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                )
                answer_enc = tokenizer(
                    record["answer"],
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                )
                label_vec = [
                    token if token != tokenizer.pad_token_id else -100
                    for token in answer_enc["input_ids"]
                ]
                input_ids.append(prompt_enc["input_ids"])
                attention_mask.append(prompt_enc["attention_mask"])
                labels.append(label_vec)
            else:
                text, _ = build_example(record)
                encoded = tokenizer(
                    text, truncation=True, max_length=max_len, padding="max_length"
                )
                ids = encoded["input_ids"]
                mask = encoded["attention_mask"]
                prompt_ids = tokenizer(
                    record["prompt"], truncation=True, max_length=max_len
                )["input_ids"]
                label_vec = ids.copy()
                for idx in range(min(len(prompt_ids), len(label_vec))):
                    label_vec[idx] = -100
                label_vec = [
                    token if token != tokenizer.pad_token_id else -100
                    for token in label_vec
                ]
                input_ids.append(ids)
                attention_mask.append(mask)
                labels.append(label_vec)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data.items()}


def tokenize_with_prompt_mask(records, tokenizer, max_len=512, *, is_encoder_decoder=False):
    return PromptMaskedDataset(
        records, tokenizer, max_len, is_encoder_decoder=is_encoder_decoder
    )


# -------------------- Model helpers --------------------


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _get_model_cls(model_id: str):
    config = AutoConfig.from_pretrained(model_id)
    if getattr(config, "is_encoder_decoder", False):
        return AutoModelForSeq2SeqLM, config
    return AutoModelForCausalLM, config


def load_fp_model(model_id: str, dtype: torch.dtype | None = None):
    model_cls, config = _get_model_cls(model_id)
    if dtype is None:
        use_bf16 = resolve_bf16(None, emit_warning=False)
        dtype = torch.bfloat16 if use_bf16 else torch.float32
    return model_cls.from_pretrained(model_id, config=config, dtype=dtype)


def load_4bit_model(model_id: str):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_cls, config = _get_model_cls(model_id)
    return model_cls.from_pretrained(
        model_id,
        config=config,
        quantization_config=quant_config,
        device_map="auto",
    )


def count_trainable_params(model) -> Tuple[int, float]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    pct = (100.0 * trainable / total) if total else 0.0
    return trainable, pct


def reset_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def peak_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def bf16_available() -> bool:
    """Return ``True`` when the current runtime supports bfloat16 training."""

    if torch.cuda.is_available() and is_torch_bf16_gpu_available():
        return True
    try:
        if is_torch_bf16_cpu_available():
            return True
    except TypeError:
        # Older versions may not expose the CPU helper yet.
        pass
    return False


def resolve_bf16(requested: bool | None, *, emit_warning: bool = True) -> bool:
    """Decide whether to enable bfloat16 based on the hardware and user request."""

    if requested is True:
        if not bf16_available():
            raise ValueError(
                "bf16 precision was requested but is not supported by the current runtime."
            )
        return True
    if requested is False:
        return False

    if bf16_available():
        return True

    if emit_warning:
        logger.info("bfloat16 not available; falling back to float32 precision.")
    return False


# -------------------- Layer indexing --------------------


def _resolve_attr_path(model, path: str):
    obj = model
    for attr in path.split("."):
        if not hasattr(obj, attr):
            return None
        obj = getattr(obj, attr)
    return obj


def get_num_layers_and_attr(model) -> Tuple[int, str]:
    for path in ("model.layers", "transformer.h"):
        obj = _resolve_attr_path(model, path)
        if obj is not None and hasattr(obj, "__len__"):
            return len(obj), path
    for path in ("encoder.block",):
        obj = _resolve_attr_path(model, path)
        if obj is not None and hasattr(obj, "__len__"):
            return len(obj), path
    raise ValueError("Could not locate transformer layers in model")


def middle_third_indices(n_layers: int) -> range:
    start = n_layers // 3
    end = math.ceil(2 * n_layers / 3)
    return range(start, end)


def last_n_indices(n_layers: int, n: int) -> range:
    return range(max(0, n_layers - n), n_layers)


# -------------------- PEFT helpers --------------------


LoRA_TARGETS_ATT_MLP = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def apply_lora_everywhere(
    model, r=16, alpha=32, dropout=0.05, targets=LoRA_TARGETS_ATT_MLP
):
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def freeze_lora_outside(model, allowed: set[int]):
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        match = _LAYER_IDX_RE.search(name)
        if not match:
            continue
        layer_idx = int(match.group(1))
        if layer_idx not in allowed:
            param.requires_grad = False


def unfreeze_lm_head(model):
    if hasattr(model, "lm_head"):
        for param in model.lm_head.parameters():
            param.requires_grad = True


# -------------------- Training wrapper --------------------


@dataclass
class TrainConfig:
    model_id: str
    train_path: str
    valid_path: str
    out_dir: str
    max_length: int = 512
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: float = 1.0
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
    max_steps: int | None = None
    overwrite_output_dir: bool = False
    evaluation_strategy: str = "no"
    eval_steps: int | None = None
    load_best_model_at_end: bool = False
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None
    early_stopping_patience: int | None = None
    early_stopping_threshold: float = 0.0
    train_limit: int | None = None
    valid_limit: int | None = None


def make_dataset(
    tokenizer,
    train_path,
    valid_path,
    max_length,
    *,
    is_encoder_decoder: bool = False,
    train_limit: int | None = None,
    valid_limit: int | None = None,
):
    train_records = load_jsonl_pair(train_path) if train_path is not None else []
    valid_records = load_jsonl_pair(valid_path) if valid_path is not None else []
    if train_limit is not None:
        train_records = train_records[:train_limit]
    if valid_limit is not None:
        valid_records = valid_records[:valid_limit]
    train_dataset = tokenize_with_prompt_mask(
        train_records,
        tokenizer,
        max_len=max_length,
        is_encoder_decoder=is_encoder_decoder,
    )
    valid_dataset = tokenize_with_prompt_mask(
        valid_records,
        tokenizer,
        max_len=max_length,
        is_encoder_decoder=is_encoder_decoder,
    )
    return train_dataset, valid_dataset


def build_trainer(
    model,
    tokenizer,
    train_ds,
    valid_ds,
    cfg: TrainConfig,
    *,
    report_to: str = "none",
):
    """Construct a :class:`~transformers.Trainer` for the provided inputs."""

    # Use a no-op collator so that the ``labels`` vector produced by
    # ``PromptMaskedDataset`` is preserved. ``DataCollatorForLanguageModeling``
    # would rebuild ``labels`` from ``input_ids`` and discard the prompt mask,
    # effectively training the model to copy its input instead of predicting the
    # answer tokens.
    collator = default_data_collator
    training_args_sig = inspect.signature(TrainingArguments.__init__)
    use_bf16 = resolve_bf16(cfg.bf16)
    eval_key = (
        "eval_strategy"
        if "eval_strategy" in training_args_sig.parameters
        else "evaluation_strategy"
    )

    metric_for_best_model = cfg.metric_for_best_model
    greater_is_better = cfg.greater_is_better
    if cfg.evaluation_strategy and cfg.evaluation_strategy.lower() != "no":
        if metric_for_best_model is None:
            metric_for_best_model = "eval_loss"
        if greater_is_better is None and metric_for_best_model == "eval_loss":
            greater_is_better = False

    load_best_model_at_end = (
        cfg.load_best_model_at_end or cfg.early_stopping_patience is not None
    )

    args_kwargs = dict(
        output_dir=cfg.out_dir,
        overwrite_output_dir=cfg.overwrite_output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        label_smoothing_factor=cfg.label_smoothing_factor,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=use_bf16,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        report_to=report_to,
        load_best_model_at_end=load_best_model_at_end,
    )
    args_kwargs[eval_key] = cfg.evaluation_strategy
    if cfg.optim:
        args_kwargs["optim"] = cfg.optim
    if cfg.lr_scheduler_type:
        args_kwargs["lr_scheduler_type"] = cfg.lr_scheduler_type
    if cfg.max_steps is not None:
        args_kwargs["max_steps"] = cfg.max_steps
    if cfg.eval_steps is not None:
        args_kwargs["eval_steps"] = cfg.eval_steps
    if metric_for_best_model is not None:
        args_kwargs["metric_for_best_model"] = metric_for_best_model
    if greater_is_better is not None:
        args_kwargs["greater_is_better"] = greater_is_better

    args = TrainingArguments(**args_kwargs)
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
    )
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    return trainer, args


def _extract_loss_history(log_history):
    history = {"train": [], "eval": []}
    for entry in log_history:
        if "loss" in entry and "learning_rate" in entry:
            history["train"].append(
                {
                    "step": entry.get("step"),
                    "epoch": entry.get("epoch"),
                    "loss": entry["loss"],
                }
            )
        if "eval_loss" in entry:
            history["eval"].append(
                {
                    "step": entry.get("step"),
                    "epoch": entry.get("epoch"),
                    "loss": entry["eval_loss"],
                }
            )
    return history


def run_trainer(model, tokenizer, train_ds, valid_ds, cfg: TrainConfig, extra_kwargs=None):
    trainer, _ = build_trainer(model, tokenizer, train_ds, valid_ds, cfg)
    if cfg.early_stopping_patience is not None:
        if not valid_ds:
            raise ValueError("Early stopping requires a validation dataset.")
        if not cfg.evaluation_strategy or cfg.evaluation_strategy.lower() == "no":
            raise ValueError(
                "Early stopping requires evaluation_strategy to be enabled."
            )
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience,
                early_stopping_threshold=cfg.early_stopping_threshold,
            )
        )
    reset_vram()
    start = time.time()
    train_output = trainer.train()
    wall = time.time() - start
    vram = peak_vram_gb()
    trainable, pct = count_trainable_params(model)
    loss_history = _extract_loss_history(trainer.state.log_history)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    efficiency = {
        "trainable_params": trainable,
        "trainable_pct": pct,
        "peak_vram_gb": vram,
        "wall_time_s": wall,
        "train_metrics": train_output.metrics,
        "loss_history": loss_history,
    }
    with open(os.path.join(cfg.out_dir, "efficiency.json"), "w", encoding="utf-8") as fp:
        json.dump(efficiency, fp, indent=2)

    model.save_pretrained(cfg.out_dir)
    tokenizer.save_pretrained(cfg.out_dir)
    return efficiency

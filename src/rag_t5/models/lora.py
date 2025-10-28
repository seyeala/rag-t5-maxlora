from dataclasses import dataclass
from typing import List

from peft import LoraConfig, TaskType, get_peft_model

# MAX profile for T5 (your choice)
TARGETS_MAX = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]


@dataclass
class LoraArgs:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] | None = None

    def __post_init__(self) -> None:
        if self.target_modules is None:
            self.target_modules = TARGETS_MAX


def apply_lora(model, args: LoraArgs):
    cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=args.target_modules,
        inference_mode=False,
    )
    peft_model = get_peft_model(model, cfg)
    return peft_model

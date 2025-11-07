from dataclasses import dataclass
from pathlib import Path
from typing import List

from peft import PeftModel

# MAX profile for T5 (your choice)
TARGETS_MAX = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]


@dataclass
class LoraArgs:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] | None = None
    adapter_path: str | None = None

    def __post_init__(self) -> None:
        if self.target_modules is None:
            self.target_modules = TARGETS_MAX
        if self.adapter_path:
            self.adapter_path = str(Path(self.adapter_path))


def apply_lora(model, args: LoraArgs):
    if not args.adapter_path:
        raise ValueError(
            "adapter_path must be provided when loading a fine-tuned LoRA adapter. "
            "Pass the trained adapter directory explicitly or via the config."
        )

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found at {adapter_path}")

    peft_model = PeftModel.from_pretrained(model, adapter_path)
    return peft_model

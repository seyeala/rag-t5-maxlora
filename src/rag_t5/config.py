from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import tomllib
from pydantic import BaseModel, Field


class DataCfg(BaseModel):
    max_tokens: int = 256
    overlap: int = 48
    dedup_threshold: float = 0.95


class RetrieverCfg(BaseModel):
    embed_model: str = "intfloat/e5-small-v2"
    index_type: str = "hnsw"
    hnsw_m: int = 32
    ef_search: int = 128
    hybrid_lambda: float = 0.25


class LoraCfg(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = Field(
        default_factory=lambda: ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
    )


class TrainCfg(BaseModel):
    task: str = "sft"
    lr: float = 5e-4
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    label_smoothing: float = 0.1
    src_max_len: int = 512
    tgt_max_len: int = 128
    gradient_checkpointing: bool = True
    model_id: str | None = None


class ServeCfg(BaseModel):
    k_retrieval: int = 2
    max_new_tokens: int = 128
    temperature: float = 0.3
    top_p: float = 0.8
    host: str = "0.0.0.0"
    port: int = 8080
    adapter_dir: str | None = None


class AppCfg(BaseModel):
    base_model: str = "google/flan-t5-small"
    artifacts_dir: str = "artifacts"
    data: DataCfg = DataCfg()
    retriever: RetrieverCfg = RetrieverCfg()
    lora: LoraCfg = LoraCfg()
    train: TrainCfg = TrainCfg()
    serve: ServeCfg = ServeCfg()


def _load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def load_settings(config_path: str, override_path: Optional[str] = None) -> AppCfg:
    base = Path(config_path)
    if not base.exists():
        raise FileNotFoundError(f"Config not found: {base}")
    data = _load_toml(base)

    if override_path:
        o = Path(override_path)
        if not o.exists():
            raise FileNotFoundError(f"Override not found: {o}")
        over = _load_toml(o)
        for k, v in over.items():
            if isinstance(v, dict) and k in data and isinstance(data[k], dict):
                data[k].update(v)
            else:
                data[k] = v

    train_data = dict(data.get("train", {}))
    if not train_data.get("model_id"):
        train_data["model_id"] = data.get("base_model", "google/flan-t5-small")

    return AppCfg(
        base_model=data.get("base_model", "google/flan-t5-small"),
        artifacts_dir=data.get("artifacts_dir", "artifacts"),
        data=DataCfg(**data.get("data", {})),
        retriever=RetrieverCfg(**data.get("retriever", {})),
        lora=LoraCfg(**data.get("lora", {})),
        train=TrainCfg(**train_data),
        serve=ServeCfg(**data.get("serve", {})),
    )

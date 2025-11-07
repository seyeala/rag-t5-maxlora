#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-v2}"  # v1|v2|v3

if python - "${CUDA_VISIBLE_DEVICES:-}" <<'PY'
import sys
try:
    import torch
except Exception:
    sys.exit(1)
sys.exit(0 if torch.cuda.is_available() else 1)
PY
then
  GPU_AVAILABLE=1
else
  GPU_AVAILABLE=0
fi

DEFAULT_MODEL="deepseek-ai/DeepSeek-V2-Lite-Chat"
if [[ "${GPU_AVAILABLE}" -eq 0 ]]; then
  DEFAULT_MODEL="google/flan-t5-small"
  if [[ "${VARIANT}" == "v2" ]]; then
    echo "[train_variant] No GPU detected; switching to CPU-friendly v3 tiny LoRA variant." >&2
    VARIANT="v3"
  fi
fi

MODEL_ID="${MODEL_ID:-${DEFAULT_MODEL}}"
MAXLEN="${MAXLEN:-512}"
EPOCHS="${EPOCHS:-1}"
BS="${BS:-1}"
ACCUM="${ACCUM:-16}"
MAX_STEPS="${MAX_STEPS:-}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
VALID_LIMIT="${VALID_LIMIT:-}"

EXTRA_ARGS=()
if [[ -n "${MAX_STEPS}" ]]; then
  EXTRA_ARGS+=("--max_steps" "${MAX_STEPS}")
fi
if [[ -n "${TRAIN_LIMIT}" ]]; then
  EXTRA_ARGS+=("--train_limit" "${TRAIN_LIMIT}")
fi
if [[ -n "${VALID_LIMIT}" ]]; then
  EXTRA_ARGS+=("--valid_limit" "${VALID_LIMIT}")
fi

if python - "${PYTHONPATH:-}" <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("datasets") is None:
    sys.exit(1)
PY
then
  python -m src.data.prepare_alpaca
else
  if [[ -f data/processed/alpaca_train.jsonl && -f data/processed/alpaca_valid.jsonl ]]; then
    echo "[train_variant] datasets library missing; using existing processed Alpaca split." >&2
  else
    echo "[train_variant] datasets library missing and processed data not found." >&2
    echo "[train_variant] Install 'datasets' or provide Alpaca splits in data/processed/." >&2
    exit 1
  fi
fi

case "${VARIANT}" in
  v1)
    CMD=(
      python -m src.train.variant_middle_ft
      --model_id "${MODEL_ID}"
      --train_path data/processed/alpaca_train.jsonl
      --valid_path data/processed/alpaca_valid.jsonl
      --out_dir outputs/v1_middle_ft
      --max_length "${MAXLEN}" --epochs "${EPOCHS}" --bs "${BS}" --accum "${ACCUM}" --lr 1e-5
    )
    CMD+=("${EXTRA_ARGS[@]}")
    "${CMD[@]}"
    ;;
  v2)
    CMD=(
      python -m src.train.variant_qlora_middle
      --model_id "${MODEL_ID}"
      --train_path data/processed/alpaca_train.jsonl
      --valid_path data/processed/alpaca_valid.jsonl
      --out_dir outputs/v2_qlora_middle
      --max_length "${MAXLEN}" --epochs "${EPOCHS}" --bs "${BS}" --accum "${ACCUM}" --lr 2e-4
    )
    CMD+=("${EXTRA_ARGS[@]}")
    "${CMD[@]}"
    ;;
  v3)
    CMD=(
      python -m src.train.variant_tiny_baseline
      --model_id "${MODEL_ID}"
      --train_path data/processed/alpaca_train.jsonl
      --valid_path data/processed/alpaca_valid.jsonl
      --out_dir outputs/v3_tiny_last2_lora
      --max_length "${MAXLEN}" --epochs "${EPOCHS}" --bs "${BS}" --accum "${ACCUM}" --lr 2e-4 --last_n 2
    )
    CMD+=("${EXTRA_ARGS[@]}")
    "${CMD[@]}"
    ;;
  *)
    echo "Use v1|v2|v3" >&2
    exit 1
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-v2}"  # v1|v2|v3
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V2-Lite-Chat}"
MAXLEN="${MAXLEN:-512}"
EPOCHS="${EPOCHS:-1}"
BS="${BS:-1}"
ACCUM="${ACCUM:-16}"

python -m src.data.prepare_alpaca

case "${VARIANT}" in
  v1)
    python -m src.train.variant_middle_ft \
      --model_id "${MODEL_ID}" \
      --train_path data/processed/alpaca_train.jsonl \
      --valid_path data/processed/alpaca_valid.jsonl \
      --out_dir outputs/v1_middle_ft \
      --max_length "${MAXLEN}" --epochs "${EPOCHS}" --bs "${BS}" --accum "${ACCUM}" --lr 1e-5
    ;;
  v2)
    python -m src.train.variant_qlora_middle \
      --model_id "${MODEL_ID}" \
      --train_path data/processed/alpaca_train.jsonl \
      --valid_path data/processed/alpaca_valid.jsonl \
      --out_dir outputs/v2_qlora_middle \
      --max_length "${MAXLEN}" --epochs "${EPOCHS}" --bs "${BS}" --accum "${ACCUM}" --lr 2e-4
    ;;
  v3)
    python -m src.train.variant_tiny_baseline \
      --model_id "${MODEL_ID}" \
      --train_path data/processed/alpaca_train.jsonl \
      --valid_path data/processed/alpaca_valid.jsonl \
      --out_dir outputs/v3_tiny_last2_lora \
      --max_length "${MAXLEN}" --epochs "${EPOCHS}" --bs "${BS}" --accum "${ACCUM}" --lr 2e-4 --last_n 2
    ;;
  *)
    echo "Use v1|v2|v3" >&2
    exit 1
    ;;
esac

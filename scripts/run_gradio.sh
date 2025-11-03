#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-outputs/v2_qlora_middle}"
PORT="${PORT:-7860}"
python -m src.apps.gradio_chat "${MODEL_DIR}" --port "${PORT}"

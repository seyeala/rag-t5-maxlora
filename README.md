# RAG-T5-MaxLoRA

RAG-T5-MaxLoRA is a research codebase for LoRA/QLoRA-style fine-tuning and evaluation. The tested training and inference paths support both encoder-decoder models such as FLAN-T5 and decoder-only causal models.

Architecture-aware defaults:
- FLAN-T5/T5: `SEQ_2_SEQ_LM` with `q, k, v, o, wi_0, wi_1, wo`.
- Llama-like causal models: `CAUSAL_LM` with `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.

See `doc/architecture-aware-lora.md` for implementation details.

## Setup

Minimal editable install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

For the regression suite:
``bash
pip install -e '.[dev]'
pytest
```

GPU training requires a CUDA-enabled PyTorch installation. The Gradio demo additionally requires `gradio`; `sentencepiece` may be needed by selected tokenizers.

## Data

Prepare the full Alpaca split:

```bash
python -m data.prepare_alpaca
```

Generated files are written to ignored `data/generated/`. The small files under `data/processed/` are tracked samples and are not overwritten.

## Train

```bash
make v1
make v2
make v3
```

The wrapper `scripts/train_variant.sh` supports `MODEL_ID`, `DATA_DIR`, `TRAIN_PATH`, `VALID_PATH`, `MAX_STEPS`, `TRAIN_LIMIT`, `VALID_LIMIT`, `EPOCHS`, `BS`, and `ACCUM`.
FLAN-T5 smoke example:

```bash
MODEL_ID=google/flan-t5-small \
MAX_STEPS=5 TRAIN_LIMIT=50 VALID_LIMIT=10 \
EPOCHS=1 BS=1 ACCUM=1 \
bash scripts/train_variant.sh v3
```

The saved PEFT adapter includes the trainable LM head when the model exposes one.

## Evaluate adapters or full models

Instruction evaluation accepts either a standalone model directory/model ID or a local PEFT adapter directory:

```bash
python -m eval.eval_instruction \
  --model_dir outputs/v3_tiny_last2_lora \
  --valid_path data/processed/alpaca_valid.jsonl \
  --limit 10
```

`src.eval.eval_sst2` uses the same architecture-aware loader for sentiment evaluation.
## Merge an adapter into a standalone model

Validated FLAN-T5 and causal adapters can be merged into their base model:

```bash
python -m rag_t5.models.export \
  outputs/v3_tiny_last2_lora \
  outputs/v3_merged
```

The merged directory contains a standalone Transformers model plus tokenizer files. T5/FLAN-T5 may emit a PEFT warning that input/output embeddings become untied during merge; deterministic pre/post-merge token output was verified for the tested FLAN-T5 path.

## Gradio demo

Install Gradio if it is not already present:

```bash
pip install gradio
python -m apps.gradio_chat outputs/v3_tiny_last2_lora
```

The demo accepts both tested PEFT adapter directories and standalone model directories. It supports seq2seq and causal generation paths.
## Tests and CI

The CPU regression suite runs with:

```bash
pip install -e '.[dev]'
pytest
```

GitHub Actions runs the suite on Python 3.11 and Python 3.12. Unit tests do not download models or datasets.

## Repository hygiene

Do not commit generated datasets, adapters, merged models, logs, virtual environments, or workstation-specific test plans. Generated training data belongs under ignored `data/generated/`; large persistent artifacts should remain outside the repository.

# RAG-T5-MaxLoRA

RAG-T5-MaxLoRA is a small research codebase for retrieval-augmented generation and LoRA/QLoRA-style fine-tuning experiments. The default configuration uses `google/flan-t5-small`, while the training scripts can also run decoder-only causal language models when the selected model supports that path.

The current training helpers choose LoRA target modules and PEFT task type from the model architecture:

- Encoder-decoder models such as FLAN-T5 use T5 module names and `SEQ_2_SEQ_LM`.
- Decoder-only causal models use projection module names such as `q_proj` and `CAUSAL_LM`.

See [`doc/architecture-aware-lora.md`](doc/architecture-aware-lora.md) for details and smoke-test commands.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install datasets accelerate bitsandbytes gradio sentencepiece
```

Use a CUDA-enabled PyTorch installation for GPU training.

## Prepare data
```bash
python -m src.data.prepare_alpaca
```

This writes the full generated Alpaca split under `data/generated/`. The small files under `data/processed/` are tracked sample data and are left unchanged.

## Train variants

```bash
make v1
make v2
make v3
```

The helper script is `scripts/train_variant.sh`. It uses `data/generated/` by default and prepares that directory automatically when needed. It supports `MODEL_ID`, `DATA_DIR`, `TRAIN_PATH`, `VALID_PATH`, `MAX_STEPS`, `TRAIN_LIMIT`, `VALID_LIMIT`, `EPOCHS`, `BS`, and `ACCUM` environment overrides.

Example smoke test with FLAN-T5:

```bash
MODEL_ID=google/flan-t5-small \
MAX_STEPS=5 TRAIN_LIMIT=50 VALID_LIMIT=10 \
EPOCHS=1 BS=1 ACCUM=1 \
bash scripts/train_variant.sh v3
```
## Evaluate

```bash
make eval_v1
make eval_v2
make eval_v3
```

## Demo

```bash
make demo_v2
```

or run directly:

```bash
python -m src.apps.gradio_chat outputs/v2_qlora_middle
```

## Repository hygiene

Do not commit generated adapters, model outputs, logs, local virtual environments, or workstation-specific test plans. Keep large local artifacts outside the repo, for example under a separate persistent storage directory.

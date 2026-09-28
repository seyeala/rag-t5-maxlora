# Architecture-aware LoRA training

The training helpers support both encoder-decoder models and decoder-only causal language models. The LoRA setup is selected from the loaded model configuration rather than assuming one family of module names for every model.

## Why this matters

FLAN-T5 is an encoder-decoder model. Its attention and feed-forward module names are different from Llama-like causal models. Applying causal-model LoRA targets such as `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` to FLAN-T5 fails because those modules do not exist in T5.

The helper in `src/train/common.py` now detects `model.config.is_encoder_decoder` and chooses the matching PEFT task type and target modules.

## Defaults by architecture

For encoder-decoder models such as `google/flan-t5-small`:

```text
PEFT task type: SEQ_2_SEQ_LM
LoRA targets:  q, k, v, o, wi_0, wi_1, wo
```

For decoder-only causal models:
```text
PEFT task type: CAUSAL_LM
LoRA targets:  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

You can override targets explicitly by passing `targets=` into `apply_lora_everywhere()` or `lora_targets=` into the high-level training config.

## A minimal FLAN-T5 smoke test

Prepare the Alpaca prompt/answer data:

```bash
python -m src.data.prepare_alpaca
```

Run a capped v3 smoke test:

```bash
MODEL_ID=google/flan-t5-small \
MAX_STEPS=5 TRAIN_LIMIT=50 VALID_LIMIT=10 \
EPOCHS=1 BS=1 ACCUM=1 \
bash scripts/train_variant.sh v3
```
This should produce adapter files under `outputs/v3_tiny_last2_lora`, including `adapter_model.safetensors`, `adapter_config.json`, tokenizer files, and `efficiency.json`.

## Programmatic training

The high-level helper also uses the architecture-aware behavior:

```python
from rag_t5.train.trainer import TrainConfig, train

cfg = TrainConfig(
    model_id="google/flan-t5-small",
    train_path="data/processed/alpaca_train.jsonl",
    valid_path="data/processed/alpaca_valid.jsonl",
    out_dir="outputs/flan_t5_lora_smoke",
    max_steps=5,
    train_limit=50,
    valid_limit=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
)
train(cfg)
```
## Notes and limits

- A smoke test verifies that the training path runs and writes adapter artifacts. It does not measure model quality.
- The v3 script still unfreezes the LM head after applying LoRA. As a result, trainable parameters include LoRA adapters plus the LM head.
- Keep generated model artifacts and local logs outside version control unless you intentionally publish them elsewhere.
- When using a new model family, confirm the target modules exist before launching a long run.

## Relevant source files

- `src/train/common.py`: model loading, target selection, PEFT setup, trainer construction.
- `src/train/variant_tiny_baseline.py`: v3 tiny LoRA training entrypoint.
- `src/rag_t5/train/trainer.py`: high-level training helper for notebooks and scripts.

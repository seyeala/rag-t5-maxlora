PY=python

MAXLEN ?= 512
EPOCHS ?= 1
BS ?= 1
ACCUM ?= 16

prep:
	$(PY) -m src.data.prepare_alpaca

v1:
	bash scripts/train_variant.sh v1

v2:
	bash scripts/train_variant.sh v2

v3:
	bash scripts/train_variant.sh v3

eval_v1:
	$(PY) -m src.eval.eval_instruction --model_dir outputs/v1_middle_ft

eval_v2:
	$(PY) -m src.eval.eval_instruction --model_dir outputs/v2_qlora_middle

eval_v3:
	$(PY) -m src.eval.eval_instruction --model_dir outputs/v3_tiny_last2_lora

demo_v2:
	$(PY) -m src.apps.gradio_chat outputs/v2_qlora_middle

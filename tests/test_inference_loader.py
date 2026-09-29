from pathlib import Path
from types import SimpleNamespace

import rag_t5.models.inference as inference


def test_model_class_uses_architecture():
    assert inference._model_class(SimpleNamespace(is_encoder_decoder=True)) is inference.AutoModelForSeq2SeqLM
    assert inference._model_class(SimpleNamespace(is_encoder_decoder=False)) is inference.AutoModelForCausalLM


def test_adapter_directory_loads_base_then_peft(monkeypatch, tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    peft_cfg = SimpleNamespace(base_model_name_or_path="base-model")
    config = SimpleNamespace(is_encoder_decoder=True)
    tokenizer = object()
    base = SimpleNamespace()
    wrapped = SimpleNamespace(eval=lambda: None)

    monkeypatch.setattr(inference.PeftConfig, "from_pretrained", lambda path: peft_cfg)
    monkeypatch.setattr(inference.AutoConfig, "from_pretrained", lambda path: config)
    monkeypatch.setattr(inference.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(inference.AutoModelForSeq2SeqLM, "from_pretrained", lambda *a, **k: base)
    monkeypatch.setattr(inference.PeftModel, "from_pretrained", lambda model, path: wrapped)

    assert inference.load_inference_model(str(tmp_path)) == (tokenizer, wrapped)

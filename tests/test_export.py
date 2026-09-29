from types import SimpleNamespace

import rag_t5.models.export as export


def test_export_requires_adapter_config(tmp_path):
    try:
        export.export_merged_adapter(str(tmp_path), str(tmp_path / "out"))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing adapter_config.json should fail")


def test_export_merges_and_saves(monkeypatch, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    output = tmp_path / "merged"
    peft_config = SimpleNamespace(base_model_name_or_path="base")
    config = SimpleNamespace(is_encoder_decoder=True)
    saved = []
    tokenizer = SimpleNamespace(save_pretrained=lambda path: saved.append(("tokenizer", path)))
    merged = SimpleNamespace(save_pretrained=lambda path: saved.append(("model", path)))
    wrapped = SimpleNamespace(merge_and_unload=lambda: merged)
    base = object()
    model_cls = SimpleNamespace(from_pretrained=lambda *a, **k: base)

    monkeypatch.setattr(export.PeftConfig, "from_pretrained", lambda path: peft_config)
    monkeypatch.setattr(export.AutoConfig, "from_pretrained", lambda path: config)
    monkeypatch.setattr(export.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(export, "_model_class", lambda cfg: model_cls)
    monkeypatch.setattr(export.PeftModel, "from_pretrained", lambda model, path: wrapped)

    assert export.export_merged_adapter(str(adapter), str(output)) == output
    assert ("model", output) in saved
    assert ("tokenizer", output) in saved

from types import SimpleNamespace

import torch

import src.eval.eval_instruction as evaluation


def test_generated_tokens_seq2seq_uses_full_output():
    model = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=True))
    output = torch.tensor([[7, 8, 9]])
    inputs = torch.tensor([[1, 2]])
    assert evaluation._generated_tokens(model, output, inputs).tolist() == [7, 8, 9]


def test_generated_tokens_causal_removes_prompt():
    model = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False))
    output = torch.tensor([[1, 2, 7, 8]])
    inputs = torch.tensor([[1, 2]])
    assert evaluation._generated_tokens(model, output, inputs).tolist() == [7, 8]


def test_load_model_selects_seq2seq(monkeypatch):
    config = SimpleNamespace(is_encoder_decoder=True)
    tokenizer = object()
    model = SimpleNamespace(eval=lambda: None)
    monkeypatch.setattr(evaluation.AutoConfig, "from_pretrained", lambda path: config)
    monkeypatch.setattr(evaluation.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(evaluation.AutoModelForSeq2SeqLM, "from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr(evaluation.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrong model class")))
    assert evaluation._load_model("fake") == (tokenizer, model)

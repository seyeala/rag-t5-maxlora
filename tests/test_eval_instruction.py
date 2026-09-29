from types import SimpleNamespace

import torch

import eval.eval_instruction as evaluation


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


def test_load_model_delegates_to_inference_loader(monkeypatch):
    expected = (object(), object())
    monkeypatch.setattr(evaluation, "load_inference_model", lambda path: expected)
    assert evaluation._load_model("fake") == expected

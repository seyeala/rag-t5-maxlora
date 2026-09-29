from types import SimpleNamespace

import torch

import apps.gradio_chat as chat


class Batch(dict):
    def to(self, device):
        return self


class Tokenizer:
    def __call__(self, text, return_tensors=None):
        return Batch(input_ids=torch.tensor([[1, 2]]))

    def decode(self, tokens, skip_special_tokens=True):
        return "decoded:" + ",".join(str(x) for x in tokens.tolist())


def test_chat_fn_seq2seq_and_causal(monkeypatch):
    monkeypatch.setattr(chat, "tokenizer", Tokenizer())

    seq = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=True), device="cpu",
                          generate=lambda **kwargs: torch.tensor([[7, 8]]))
    monkeypatch.setattr(chat, "model", seq)
    assert chat.chat_fn("do", "") == "decoded:7,8"

    causal = SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False), device="cpu",
                             generate=lambda **kwargs: torch.tensor([[1, 2, 7, 8]]))
    monkeypatch.setattr(chat, "model", causal)
    assert chat.chat_fn("do", "") == "decoded:7,8"


def test_build_app_delegates_loader(monkeypatch):
    expected=(Tokenizer(), SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=True)))
    monkeypatch.setattr(chat, "_load", lambda path: expected)
    demo=chat.build_app("adapter")
    assert chat.tokenizer is expected[0]
    assert chat.model is expected[1]
    assert demo is not None

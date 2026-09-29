from types import SimpleNamespace

import torch

import rag_t5.eval.sst2 as sst2


def test_parse_label():
    assert sst2._parse_label("positive") == "positive"
    assert sst2._parse_label("negative") == "negative"
    assert sst2._parse_label("pos") == "positive"
    assert sst2._parse_label("neg") == "negative"


def test_evaluate_uses_shared_loader_and_seq2seq_decode(monkeypatch, tmp_path):
    data = tmp_path / "sst2.jsonl"
    data.write_text('{"prompt":"good","answer":"positive"}\n')
    model = SimpleNamespace(
        config=SimpleNamespace(is_encoder_decoder=True),
        device="cpu",
        generate=lambda **kwargs: torch.tensor([[5, 6]]),
    )

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def __call__(self, text, return_tensors=None):
            return Batch(input_ids=torch.tensor([[1, 2]]))
        def decode(self, tokens, skip_special_tokens=True):
            assert tokens.tolist() == [5, 6]
            return "positive"

    monkeypatch.setattr(sst2, "load_inference_model", lambda path: (Tokenizer(), model))
    assert sst2.evaluate("fake", str(data), limit=1) == {"n": 1, "accuracy": 1.0}

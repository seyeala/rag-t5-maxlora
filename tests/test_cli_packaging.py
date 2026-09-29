import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "cli.chunk",
        "cli.gen_synth",
        "cli.generate",
        "cli.ingest",
        "cli.smoke_model",
        "cli.train",
    ],
)
def test_cli_modules_import_without_optional_retrieval_dependency(module):
    sys.modules.pop(module, None)
    assert importlib.import_module(module) is not None


def test_generate_reports_retrieval_extra_when_faiss_missing(monkeypatch, tmp_path):
    generate = importlib.import_module("cli.generate")
    index = tmp_path / "index.faiss"
    index.write_bytes(b"placeholder")

    real_import = __import__

    def blocked_import(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    with pytest.raises(RuntimeError, match="retrieval.*extra"):
        generate._load_faiss_index(index)

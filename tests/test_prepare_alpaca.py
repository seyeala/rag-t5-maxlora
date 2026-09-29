import json

import src.data.prepare_alpaca as prep


def test_build_prompt_with_and_without_input():
    plain = prep.build_prompt({"instruction": "Do it", "input": ""})
    with_input = prep.build_prompt({"instruction": "Do it", "input": "Context"})

    assert "### Input:" not in plain
    assert "### Response:" in plain
    assert "### Input:\nContext" in with_input


def test_prepare_writes_requested_generated_directory(monkeypatch, tmp_path):
    rows = [
        {"instruction": f"task {i}", "input": "", "output": f"answer {i}"}
        for i in range(220)
    ]
    monkeypatch.setattr(prep, "load_dataset", lambda name: {"train": rows})
    monkeypatch.setattr(prep.random, "shuffle", lambda records: None)

    out_dir = tmp_path / "generated"
    prep.main(out_dir=str(out_dir), split_ratio=0.1)
    train_path = out_dir / "alpaca_train.jsonl"
    valid_path = out_dir / "alpaca_valid.jsonl"
    assert train_path.exists()
    assert valid_path.exists()

    train_rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    valid_rows = [json.loads(line) for line in valid_path.read_text().splitlines()]

    assert len(valid_rows) == 200
    assert len(train_rows) == 20
    assert set(train_rows[0]) == {"prompt", "answer"}
    assert not (tmp_path / "data" / "processed" / "alpaca_train.jsonl").exists()

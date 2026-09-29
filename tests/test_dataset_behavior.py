import json

from train.common import make_dataset


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, truncation=True, max_length=8, padding=None):
        ids = [min(ord(ch), 255) for ch in text][:max_length]
        if padding == "max_length":
            ids = ids + [0] * (max_length - len(ids))
        return {"input_ids": ids, "attention_mask": [1 if token else 0 for token in ids]}


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record) + "\n")


def test_make_dataset_limits_and_seq2seq_labels(tmp_path):
    records = [
        {"prompt": "p1", "answer": "a1"},
        {"prompt": "p2", "answer": "a2"},
        {"prompt": "p3", "answer": "a3"},
    ]
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    write_jsonl(train, records)
    write_jsonl(valid, records)
    train_ds, valid_ds = make_dataset(
        FakeTokenizer(),
        str(train),
        str(valid),
        8,
        is_encoder_decoder=True,
        train_limit=2,
        valid_limit=1,
    )

    assert len(train_ds) == 2
    assert len(valid_ds) == 1
    item = train_ds[0]
    assert item["input_ids"][:2] == [ord("p"), ord("1")]
    assert item["labels"][:2] == [ord("a"), ord("1")]
    assert all(token == -100 for token in item["labels"][2:])


def test_causal_labels_mask_prompt(tmp_path):
    path = tmp_path / "data.jsonl"
    write_jsonl(path, [{"prompt": "ab", "answer": "cd"}])

    train_ds, _ = make_dataset(FakeTokenizer(), str(path), None, 8)
    labels = train_ds[0]["labels"]

    assert labels[:2] == [-100, -100]
    assert labels[2:4] == [ord("c"), ord("d")]

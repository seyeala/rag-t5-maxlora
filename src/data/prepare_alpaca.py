import json
import random
from pathlib import Path

from datasets import load_dataset


random.seed(13)


def build_prompt(example):
    instruction = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    if inp:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{inp}\n\n"
            "### Response:\n"
        )
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def main(out_dir="data/processed", split_ratio=0.1):
    dataset = load_dataset("yahma/alpaca-cleaned")["train"]
    records = []
    for example in dataset:
        prompt = build_prompt(example)
        answer = (example.get("output") or "").strip()
        if not answer:
            continue
        records.append({"prompt": prompt, "answer": answer})

    random.shuffle(records)
    val_count = max(200, int(len(records) * split_ratio))
    val_records = records[:val_count]
    train_records = records[val_count:]

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "alpaca_train.jsonl"
    valid_path = output_dir / "alpaca_valid.jsonl"

    with train_path.open("w", encoding="utf-8") as fp:
        for record in train_records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    with valid_path.open("w", encoding="utf-8") as fp:
        for record in val_records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(train_records)} train / {len(val_records)} valid records to {output_dir}"
    )


if __name__ == "__main__":
    main()

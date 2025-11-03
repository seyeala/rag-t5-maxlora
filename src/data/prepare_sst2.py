import json
from pathlib import Path

from datasets import load_dataset


LABEL_TEXT = {0: "negative", 1: "positive"}


def main(out_dir="data/processed"):
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("glue", "sst2")
    for split in ("train", "validation"):
        path = output_dir / f"sst2_{split}.jsonl"
        with path.open("w", encoding="utf-8") as fp:
            for example in dataset[split]:
                prompt = (
                    f"Sentence: {example['sentence']}\n"
                    "Label (positive/negative):"
                )
                fp.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "answer": LABEL_TEXT[int(example["label"])],
                        }
                    )
                    + "\n"
                )
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

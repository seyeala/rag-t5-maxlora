import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _parse_label(text: str) -> str:
    lowered = text.lower()
    if "positive" in lowered and "negative" not in lowered:
        return "positive"
    if "negative" in lowered and "positive" not in lowered:
        return "negative"
    if lowered.startswith("pos"):
        return "positive"
    if lowered.startswith("neg"):
        return "negative"
    return "positive"


def evaluate(model_dir, path="data/processed/sst2_validation.jsonl", limit=500):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    model.eval()

    gold, predictions = [], []
    with open(path, encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            if limit and idx >= limit:
                break
            example = json.loads(line)
            inputs = tokenizer(example["prompt"], return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=4, do_sample=False
                )
            generated = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            predictions.append(_parse_label(generated))
            gold.append(example["answer"])

    correct = sum(1 for pred, target in zip(predictions, gold) if pred == target)
    accuracy = correct / len(gold) if gold else 0.0
    return {"n": len(gold), "accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--path", default="data/processed/sst2_validation.jsonl")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    metrics = evaluate(args.model_dir, path=args.path, limit=args.limit)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
import re
import string

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = sum(token in ref_tokens for token in pred_tokens) / len(pred_tokens)
    recall = sum(token in pred_tokens for token in ref_tokens) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(model_dir, valid_path, max_new_tokens=128, limit=200):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    model.eval()

    examples = []
    with open(valid_path, encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            if limit and idx >= limit:
                break
            examples.append(json.loads(line))

    em_scores, f1_scores = [], []
    for example in examples:
        prompt = example["prompt"]
        answer = example["answer"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        em_scores.append(1.0 if _normalize(generated) == _normalize(answer) else 0.0)
        f1_scores.append(_f1(generated, answer))

    return {
        "n": len(em_scores),
        "EM": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "F1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument(
        "--valid_path", default="data/processed/alpaca_valid.jsonl"
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    metrics = evaluate(
        model_dir=args.model_dir,
        valid_path=args.valid_path,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

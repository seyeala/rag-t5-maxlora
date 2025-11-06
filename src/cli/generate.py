from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import faiss
import torch
from transformers import AutoModel, AutoTokenizer

from rag_t5.config import load_settings
from rag_t5.data.ingest import normalize_text
from rag_t5.models.loader import load_base
from rag_t5.models.lora import LoraArgs, apply_lora
from rag_t5.prompt import format_prompt
from rag_t5.utils.io import stream_jsonl


def _load_faiss_index(index_path: Path) -> faiss.Index:
    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index at {index_path}")
    index = faiss.read_index(str(index_path))
    return index


def _load_chunks(chunks_path: Path) -> List[dict]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file at {chunks_path}")
    return list(stream_jsonl(chunks_path))


def _embed_query(text: str, model_name: str) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = summed / lengths
    else:
        pooled = hidden.mean(dim=1)

    return pooled.detach().cpu().numpy().astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Query Generation CLI")
    parser.add_argument("--query", "-q", required=True, type=str, help="Input query text")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument(
        "--model",
        choices=["base", "lora"],
        default="base",
        help="Choose the generator variant: base T5 or LoRA-adapted",
    )
    parser.add_argument(
        "--highlight",
        "-H",
        action="store_true",
        help="Highlight query terms within retrieved context",
    )
    parser.add_argument(
        "--context_position",
        choices=["prepend", "append"],
        default="prepend",
        help="Where to place context relative to the query in the prompt",
    )
    parser.add_argument("--temperature", "-t", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p sampling threshold")
    parser.add_argument(
        "--max_new_tokens",
        "-m",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--show_chunks",
        "-s",
        action="store_true",
        help="Display retrieved chunk metadata and text",
    )
    parser.add_argument(
        "--config", "-c", type=str, default="configs/defaults.toml", help="Base config path"
    )
    parser.add_argument(
        "--override",
        "-O",
        type=str,
        default=None,
        help="Optional override config path",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="Path to a fine-tuned LoRA adapter (overrides config)",
    )
    args = parser.parse_args()

    cfg = load_settings(args.config, args.override)

    if args.top_k <= 0:
        parser.error("--top_k must be a positive integer")

    artifacts_dir = Path(cfg.artifacts_dir)
    chunks_path = artifacts_dir / "processed" / "chunks.jsonl"
    index_path = artifacts_dir / "models" / "chunks_index.faiss"

    index = _load_faiss_index(index_path)
    if cfg.retriever.index_type.lower() == "hnsw" and hasattr(index, "hnsw"):
        index.hnsw.efSearch = cfg.retriever.ef_search

    query_text = normalize_text(args.query)
    if not query_text:
        raise ValueError("Query text is empty after normalization")

    query_vector = _embed_query(query_text, cfg.retriever.embed_model)

    distances, indices = index.search(query_vector, args.top_k)

    chunks = _load_chunks(chunks_path)
    retrieved = []
    for rank, (chunk_idx, score) in enumerate(zip(indices[0], distances[0]), start=1):
        if chunk_idx < 0 or chunk_idx >= len(chunks):
            continue
        chunk = chunks[chunk_idx]
        retrieved.append({"rank": rank, "score": float(score), "chunk": chunk})

    context_texts = [item["chunk"].get("text", "") for item in retrieved]

    prompt = format_prompt(
        context_texts,
        query_text,
        highlight=args.highlight,
        position=args.context_position,
    )

    model, tokenizer, device = load_base(cfg.base_model)
    adapter_dir = args.adapter_dir or getattr(cfg.serve, "adapter_dir", None)
    if args.model == "lora":
        if adapter_dir is None:
            parser.error(
                "LoRA model requested but no adapter directory provided. "
                "Specify --adapter_dir or set serve.adapter_dir in the config."
            )
        lora_args = LoraArgs(
            r=cfg.lora.r,
            alpha=cfg.lora.alpha,
            dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            weights_path=adapter_dir,
        )
        model = apply_lora(model, lora_args)
        model.to(device)
    model.eval()

    encoded = tokenizer(prompt, return_tensors="pt").to(device)

    do_sample = args.temperature > 0.0
    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})

    with torch.no_grad():
        generated = model.generate(**encoded, **gen_kwargs)
    answer = tokenizer.decode(generated[0], skip_special_tokens=True)

    print("\nFinal Answer:\n" + answer)

    if args.show_chunks and retrieved:
        print("\nRetrieved Chunks:")
        for item in retrieved:
            chunk = item["chunk"]
            score = item["score"]
            cid = chunk.get("chunk_id", "<unknown>")
            text = chunk.get("text", "").strip()
            print(f"[{item['rank']}] {cid} (score={score:.4f})\n{text}\n")


if __name__ == "__main__":
    main()

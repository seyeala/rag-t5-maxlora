from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from rag_t5.utils.io import ensure_dir
from rag_t5.config import AppCfg


def _chunk_tokens_greedy_with_overlap(tokenizer, text: str, max_tokens: int, overlap: int) -> List[Dict]:
    """
    Packs paragraphs into chunks up to max_tokens. When a chunk is flushed,
    the next chunk starts with the last `overlap` tokens from the previous one.
    For paragraphs longer than max_tokens, they are internally split with stride=(max-overlap).
    Returns dicts with chunk_index, text, token_count, start_token, end_token.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    para_ids = [tokenizer.encode(p, add_special_tokens=False) for p in paras]

    chunks: List[Dict] = []
    cur_ids: List[int] = []
    doc_cursor = 0  # approximate token start of current buffer

    def flush():
        nonlocal cur_ids, doc_cursor
        if not cur_ids:
            return
        idx = len(chunks)
        chunk_text = tokenizer.decode(cur_ids, skip_special_tokens=True)
        length = len(cur_ids)
        rec = {
            "chunk_index": idx,
            "text": chunk_text,
            "token_count": length,
            "start_token": doc_cursor,
            "end_token": doc_cursor + length,
        }
        chunks.append(rec)
        # prepare overlap tail for the next chunk
        tail = cur_ids[-overlap:] if overlap > 0 else []
        doc_cursor = rec["end_token"] - len(tail)
        cur_ids = tail[:]  # carry forward

    for ids in para_ids:
        space_left = max_tokens - len(cur_ids)
        if len(ids) <= space_left:
            cur_ids.extend(ids)
            continue

        # flush current chunk
        flush()

        # if a single paragraph still exceeds max_tokens, split it with stride
        if len(ids) > max_tokens:
            stride = max_tokens - overlap
            for s in range(0, len(ids), stride):
                part = ids[s:s + max_tokens]
                cur_ids.extend(part)
                flush()
        else:
            cur_ids.extend(ids)

    # final flush
    flush()
    return chunks


def chunk_docs(docs_path: str | Path, chunks_path: str | Path, cfg: AppCfg) -> Dict[str, float | int | str]:
    docs_path = Path(docs_path)
    chunks_path = Path(chunks_path)
    ensure_dir(chunks_path.parent)

    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)

    n_docs = n_chunks = total_tokens = 0
    with docs_path.open("r", encoding="utf-8") as fin, chunks_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Chunk docs"):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            pieces = _chunk_tokens_greedy_with_overlap(tok, doc["text"], cfg.data.max_tokens, cfg.data.overlap)
            for piece in pieces:
                rec = {
                    "doc_id": doc["doc_id"],
                    "chunk_id": f'{doc["doc_id"]}:{piece["chunk_index"]:04d}',
                    "text": piece["text"],
                    "token_count": piece["token_count"],
                    "start_token": piece["start_token"],
                    "end_token": piece["end_token"],
                    "source_path": doc.get("source_path"),
                    "title": doc.get("title"),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks += 1
                total_tokens += piece["token_count"]
            n_docs += 1

    avg_tokens = (total_tokens / n_chunks) if n_chunks else 0.0
    return {"docs": n_docs, "chunks": n_chunks, "avg_tokens": round(avg_tokens, 2), "out": str(chunks_path)}

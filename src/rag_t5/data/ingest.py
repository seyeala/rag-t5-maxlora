from __future__ import annotations

from pathlib import Path
from typing import Dict
from hashlib import sha1
import unicodedata, re, json
from tqdm import tqdm
from rag_t5.utils.io import ensure_dir

CONTROL_WS = re.compile(r"[^\S\n]+")       # collapse spaces/tabs but keep \n
MULTI_NL   = re.compile(r"\n{3,}")         # collapse 3+ newlines to 2
SAFE_STEM  = re.compile(r"[^A-Za-z0-9_-]+")

def normalize_text(s: str) -> str:
    # Unicode normalize, drop BOM, normalize newlines, collapse whitespace
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\ufeff", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = CONTROL_WS.sub(" ", s)
    s = MULTI_NL.sub("\n\n", s)
    s = s.strip()
    return s

def iter_txt_files(in_dir: Path, recursive: bool = True):
    pattern = "**/*.txt" if recursive else "*.txt"
    for p in sorted(in_dir.glob(pattern)):
        if p.is_file():
            yield p

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def make_doc_id(path: Path, text: str) -> str:
    h = sha1(text.encode("utf-8")).hexdigest()[:8]
    stem = SAFE_STEM.sub("-", path.stem)[:32].strip("-") or "doc"
    return f"{stem}-{h}"

def ingest_dir(
    in_dir: str | Path,
    out_path: str | Path,
    index_path: str | Path | None = None,
    min_chars: int = 100,
    recursive: bool = True,
) -> Dict[str, int | str]:
    in_dir = Path(in_dir)
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    seen_hashes = set()
    n_in = n_kept = n_dupe = n_short = 0
    index: Dict[str, str] = {}

    with out_path.open("w", encoding="utf-8") as fout:
        files = list(iter_txt_files(in_dir, recursive=recursive))
        for fp in tqdm(files, desc="Ingest .txt"):
            n_in += 1
            raw = read_text(fp)
            norm = normalize_text(raw)
            if len(norm) < min_chars:
                n_short += 1
                continue
            h = sha1(norm.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                n_dupe += 1
                continue
            seen_hashes.add(h)
            doc_id = make_doc_id(fp, norm)
            rec = {
                "doc_id": doc_id,
                "source_path": str(fp.relative_to(in_dir)),
                "title": fp.stem,
                "text": norm,
                "n_chars": len(norm),
                "sha1": h,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            index[doc_id] = rec["source_path"]
            n_kept += 1

    if index_path:
        index_path = Path(index_path)
        ensure_dir(index_path.parent)
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"files_seen": n_in, "docs_written": n_kept, "duplicates": n_dupe, "too_short": n_short, "out": str(out_path)}

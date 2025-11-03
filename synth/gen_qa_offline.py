from __future__ import annotations

import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from rag_t5.utils.io import ensure_dir, stream_jsonl, write_jsonl

logger = logging.getLogger(__name__)

_OPENAI_INITIALIZED = False
_OPENAI_AVAILABLE = False
_OPENAI_MODULE = None

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CAPITAL_PHRASE_RE = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,4})\b")
PRONOUNS = {"it", "they", "he", "she", "this", "that", "these", "those", "there", "here", "its"}
LINKING_PHRASES = [
    " is ",
    " are ",
    " was ",
    " were ",
    " refers to ",
    " consists of ",
    " means ",
    " describes ",
    " defines ",
    " involves ",
    " represents ",
]


@dataclass
class QAExample:
    context: str
    question: str
    answer: str

    def as_record(self) -> dict:
        return {"src": f"{self.context}\n\n{self.question}", "tgt": self.answer}


def split_sentences(text: str) -> List[str]:
    sentences = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def clean_phrase(text: str) -> str:
    text = text.strip().strip("\"'“”‘’()[]{}")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_subject(text: str) -> Optional[str]:
    text = clean_phrase(text)
    if not text:
        return None
    lowered = text.lower()
    if lowered in PRONOUNS:
        return None
    if len(text) < 3:
        return None
    if text.count(" ") > 10:
        return None
    return text


def clean_answer(text: str, subject: Optional[str] = None) -> Optional[str]:
    text = clean_phrase(text)
    if subject:
        lowered_subject = subject.lower()
        lowered_text = text.lower()
        if lowered_text.startswith(lowered_subject):
            tail = text[len(subject):].lstrip(" ,:-")
            if tail:
                text = tail
    text = text.rstrip(".;:")
    if not text:
        return None
    return text


def parse_chunk_index(chunk_id: str) -> Optional[int]:
    if not chunk_id:
        return None
    if ":" not in chunk_id:
        return None
    try:
        return int(chunk_id.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


def extract_answer_from_sentence(sentence: str, subject: str) -> Optional[str]:
    lowered_sentence = sentence.lower()
    lowered_subject = subject.lower()
    idx = lowered_sentence.find(lowered_subject)
    if idx == -1:
        return None
    after_subject = sentence[idx + len(subject):]
    lowered_after = after_subject.lower()
    for phrase in LINKING_PHRASES:
        if lowered_after.startswith(phrase.strip()):
            answer = after_subject[len(phrase.strip()):].strip()
            return clean_answer(answer, subject)
        if lowered_after.startswith(phrase):
            answer = after_subject[len(phrase):].strip()
            return clean_answer(answer, subject)
    # No linking phrase match; fall back to tail of sentence
    tail = after_subject.strip()
    return clean_answer(tail, subject)


def find_subject_and_answer(sentences: Iterable[str], fallback_subject: Optional[str] = None) -> Optional[tuple[str, str]]:
    sentences_list = list(sentences)
    if fallback_subject:
        subject = clean_subject(fallback_subject)
        if subject:
            for sent in sentences_list:
                answer = extract_answer_from_sentence(sent, subject)
                if answer:
                    return subject, answer
            if sentences_list:
                answer = clean_answer(sentences_list[0], subject)
                if answer:
                    return subject, answer
    for sent in sentences_list:
        lowered = sent.lower()
        for phrase in LINKING_PHRASES:
            idx = lowered.find(phrase)
            if idx > 0:
                subject_candidate = clean_subject(sent[:idx])
                if not subject_candidate:
                    continue
                answer_candidate = clean_answer(sent[idx + len(phrase):], subject_candidate)
                if not answer_candidate:
                    continue
                return subject_candidate, answer_candidate
    # Fall back to a capitalised phrase
    if sentences_list:
        match = CAPITAL_PHRASE_RE.search(sentences_list[0])
        if match:
            subject_candidate = clean_subject(match.group(1))
            if subject_candidate:
                answer_candidate = extract_answer_from_sentence(sentences_list[0], subject_candidate)
                if not answer_candidate:
                    answer_candidate = clean_answer(sentences_list[0], subject_candidate)
                if answer_candidate:
                    return subject_candidate, answer_candidate
    return None


def generate_offline_example(chunk: dict) -> Optional[QAExample]:
    text = (chunk.get("text") or "").strip()
    if len(text) < 50:
        return None
    sentences = split_sentences(text)
    if not sentences:
        return None
    chunk_index = parse_chunk_index(chunk.get("chunk_id", ""))
    title = (chunk.get("title") or "").strip() or None
    fallback_subject = None
    if title and chunk_index == 0:
        fallback_subject = title
    subject_answer = find_subject_and_answer(sentences, fallback_subject)
    if not subject_answer:
        return None
    subject, answer = subject_answer
    question = build_question(subject)
    context = text
    return QAExample(context=context, question=question, answer=answer)


def build_question(subject: str, entity_type: Optional[str] = None) -> str:
    subject = subject.strip()
    if not subject:
        raise ValueError("Subject cannot be empty when building question.")
    prefix = "What"
    if entity_type == "person":
        prefix = "Who"
    elif entity_type == "location":
        prefix = "Where"
    question = f"{prefix} is {subject}?"
    if not question.endswith("?"):
        question += "?"
    return question


def _ensure_openai_client():
    global _OPENAI_INITIALIZED, _OPENAI_AVAILABLE, _OPENAI_MODULE
    if _OPENAI_INITIALIZED:
        return _OPENAI_MODULE if _OPENAI_AVAILABLE else None

    _OPENAI_INITIALIZED = True
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; using offline question generation.")
        _OPENAI_AVAILABLE = False
        return None
    try:
        import openai  # type: ignore
    except ImportError:
        logger.warning("openai package not installed; using offline question generation.")
        _OPENAI_AVAILABLE = False
        return None

    openai.api_key = api_key
    _OPENAI_MODULE = openai
    _OPENAI_AVAILABLE = True
    return openai


def maybe_generate_question_with_api(chunk: dict) -> Optional[str]:
    openai = _ensure_openai_client()
    if not openai:
        return None

    if not hasattr(openai, "ChatCompletion"):
        logger.warning("Installed openai package has no ChatCompletion client; using offline generation.")
        # Avoid repeating this warning for every chunk
        global _OPENAI_AVAILABLE, _OPENAI_MODULE
        _OPENAI_AVAILABLE = False
        _OPENAI_MODULE = None
        return None

    context = chunk.get("text", "")
    prompt = (
        "You are a helpful assistant that writes reading comprehension questions. "
        "Given the context, write a single question that can be answered directly from the text. "
        "Return the result in the format 'Q: <question>' on one line and 'A: <answer>' on another line."
    )
    try:
        response = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Context:\n{context}"},
            ],
            temperature=0.3,
            max_tokens=256,
        )
    except Exception as exc:  # pragma: no cover - API errors are non-deterministic
        logger.warning("API question generation failed: %s", exc)
        return None

    try:
        content = response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        logger.warning("Unexpected API response format; falling back to offline generation.")
        return None

    question, _answer = parse_api_output(content)
    if question:
        return question
    logger.warning("Could not parse question from API output; using offline generation.")
    return None


def parse_api_output(text: str) -> tuple[Optional[str], Optional[str]]:
    question_lines: List[str] = []
    answer_lines: List[str] = []
    mode: Optional[str] = None
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if lowered.startswith("q:"):
            mode = "q"
            question_lines.append(stripped.split(":", 1)[1].strip())
        elif lowered.startswith("a:"):
            mode = "a"
            answer_lines.append(stripped.split(":", 1)[1].strip())
        else:
            if mode == "q":
                question_lines.append(stripped)
            elif mode == "a":
                answer_lines.append(stripped)
    question = " ".join(question_lines).strip() if question_lines else None
    answer = " ".join(answer_lines).strip() if answer_lines else None
    return question or None, answer or None


def apply_api_question(example: QAExample, chunk: dict) -> QAExample:
    question = maybe_generate_question_with_api(chunk)
    if not question:
        return example
    question = question.strip()
    if not question.endswith("?"):
        question += "?"
    return QAExample(context=example.context, question=question, answer=example.answer)


def generate_sft_examples(
    chunks_path: str | Path,
    *,
    use_api: bool = False,
    seed: int | None = 13,
    min_examples: int = 100,
) -> List[QAExample]:
    chunks_path = Path(chunks_path)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunks_path}")

    rng = random.Random(seed)
    examples: List[QAExample] = []
    for chunk in stream_jsonl(chunks_path):
        example = generate_offline_example(chunk)
        if not example:
            continue
        if use_api:
            example = apply_api_question(example, chunk)
        examples.append(example)

    examples = dedupe_examples(examples)
    if not examples:
        raise ValueError("No QA examples generated from the provided chunks.")
    rng.shuffle(examples)
    if len(examples) < min_examples:
        raise ValueError(
            f"Only generated {len(examples)} examples; consider relaxing heuristics or providing more data."
        )
    return examples


def dedupe_examples(examples: List[QAExample]) -> List[QAExample]:
    seen = set()
    deduped: List[QAExample] = []
    for example in examples:
        key = (example.context, example.question)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


def save_sft_dataset(examples: Iterable[QAExample], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    records = [ex.as_record() for ex in examples]
    write_jsonl(out_path, records)
    return out_path


def preview_examples(examples: List[QAExample], n: int = 3) -> List[dict]:
    preview = []
    for example in examples[:n]:
        preview.append({
            "question": example.question,
            "answer": example.answer,
        })
    return preview


__all__ = [
    "QAExample",
    "generate_sft_examples",
    "save_sft_dataset",
    "preview_examples",
]

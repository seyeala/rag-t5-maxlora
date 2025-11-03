from __future__ import annotations

from typing import List, Sequence, Union
import re


_HIGHLIGHT_SKIP = {"the", "and", "is", "are", "of", "to", "a", "an"}


def _ensure_list(chunks: Union[str, Sequence[str]] | None) -> List[str]:
    if chunks is None:
        return []
    if isinstance(chunks, str):
        return [chunks]
    return list(chunks)


def _highlight_text(text: str, query: str) -> str:
    query_words = [
        w
        for w in re.findall(r"\w+", query.lower())
        if len(w) >= 3 and w not in _HIGHLIGHT_SKIP
    ]
    if not query_words:
        return text

    # Combine into a single regex so each span is highlighted only once.
    pattern = re.compile(
        r"\\b(" + "|".join(re.escape(w) for w in sorted(set(query_words), key=len, reverse=True)) + r")\\b",
        re.IGNORECASE,
    )

    def replacer(match: re.Match[str]) -> str:
        return f"<hl>{match.group(0)}</hl>"

    return pattern.sub(replacer, text)


def format_prompt(
    chunks: Union[str, Sequence[str]] | None,
    query: str,
    highlight: bool = False,
    position: str = "prepend",
) -> str:
    """Compose a prompt from retrieved context chunks and a query.

    Parameters
    ----------
    chunks:
        Either a single context string, an iterable of chunk strings, or ``None``.
    query:
        The user query or question to append to the prompt.
    highlight:
        When ``True``, wrap query term occurrences inside the context with ``<hl>`` markers.
    position:
        Controls whether the context should be placed ``"prepend"`` (before the query)
        or ``"append"`` (after the query).
    """

    chunk_list = _ensure_list(chunks)
    formatted: List[str] = []
    for chunk in chunk_list:
        if highlight:
            formatted.append(_highlight_text(chunk, query))
        else:
            formatted.append(chunk)

    context_text = "\n\n".join(formatted).strip()

    if position not in {"prepend", "append"}:
        raise ValueError("position must be 'prepend' or 'append'")

    if not context_text:
        return query

    if position == "prepend":
        return f"{context_text}\n\n{query}"
    return f"{query}\n\n{context_text}"

"""Data processing utilities for RAG T5."""

__all__ = ["ingest_dir", "chunk_docs"]


def ingest_dir(*args, **kwargs):
    from .ingest import ingest_dir as _impl

    return _impl(*args, **kwargs)


def chunk_docs(*args, **kwargs):
    from .chunk import chunk_docs as _impl

    return _impl(*args, **kwargs)

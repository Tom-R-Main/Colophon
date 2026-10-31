"""Compatibility entry point for exact, worker-independent measurements.

Unsafe per-chunk approximations have been retired. Whole-document parsing is
intentionally used until a streaming implementation can preserve sentence,
quotation, vocabulary and cross-boundary ngram contracts.
"""

from __future__ import annotations

from colophon.models.document import Document
from colophon.models.features import StyleProfile

CHUNK_TARGET = 50_000


def chunk_text(text: str, target_words: int = CHUNK_TARGET) -> list[str]:
    """Split only for callers needing display batches, never for measurements."""
    if target_words < 1:
        raise ValueError("target_words must be positive")
    chunks: list[str] = []
    current: list[str] = []
    word_count = 0
    for paragraph in text.split("\n\n"):
        if not paragraph.strip():
            continue
        current.append(paragraph)
        word_count += len(paragraph.split())
        if word_count >= target_words:
            chunks.append("\n\n".join(current))
            current, word_count = [], 0
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def analyze_chunked(
    doc: Document,
    *,
    analyzers: list[str],
    spacy_model: str,
    lang: str,
    chunk_size: int = CHUNK_TARGET,
    workers: int | None = None,
) -> StyleProfile:
    """Use the same exact pipeline regardless of legacy batching parameters."""
    from colophon.analysis.pipeline import analyze

    return analyze(
        doc, analyzers=analyzers, spacy_model=spacy_model,
        lang=lang, chunk_size=chunk_size, workers=workers,
    )

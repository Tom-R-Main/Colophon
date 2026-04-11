"""Document ingestion — PDF, EPUB, TXT, MD."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from colophon.models.document import Document

SUPPORTED_SUFFIXES = {".pdf", ".epub", ".txt", ".md"}


def ingest(path: Path, *, no_segment: bool = False, segment_pattern: str | None = None) -> Document:
    """Ingest a document and return a normalized Document model."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}")

    if suffix == ".pdf":
        from colophon.ingestion.pdf import ingest_pdf
        return ingest_pdf(path, no_segment=no_segment, segment_pattern=segment_pattern)
    elif suffix == ".epub":
        from colophon.ingestion.epub import ingest_epub
        return ingest_epub(path, no_segment=no_segment, segment_pattern=segment_pattern)
    else:
        from colophon.ingestion.text import ingest_text
        return ingest_text(path, no_segment=no_segment, segment_pattern=segment_pattern)

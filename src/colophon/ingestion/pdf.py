"""PDF file ingestion using pymupdf (primary) and pdfplumber (fallback)."""

from __future__ import annotations

import re
from pathlib import Path

from colophon.ingestion.normalize import normalize_text
from colophon.models.document import Document, Segment


def ingest_pdf(
    path: Path,
    *,
    no_segment: bool = False,
    segment_pattern: str | None = None,
) -> Document:
    """Ingest a PDF file, extracting and normalizing text."""
    text = _extract_with_pymupdf(path)

    # If pymupdf yields very little text, try pdfplumber
    page_count = _get_page_count(path)
    chars_per_page = len(text) / max(page_count, 1)
    if chars_per_page < 100:
        pdfplumber_text = _extract_with_pdfplumber(path)
        if len(pdfplumber_text) > len(text):
            text = pdfplumber_text

    # If still very little text, suggest OCR
    chars_per_page = len(text) / max(page_count, 1)
    if chars_per_page < 50:
        try:
            from colophon.ingestion.ocr import ocr_pdf
            ocr_path = ocr_pdf(path)
            text = _extract_with_pymupdf(ocr_path)
        except (ImportError, RuntimeError):
            pass  # OCR not available, proceed with what we have

    text = normalize_text(text)
    metadata = {"page_count": str(page_count)}

    # Extract title from PDF metadata if available
    title = _extract_title(path) or path.stem

    if no_segment:
        segments = [Segment(index=0, title=None, text=text, char_offset_start=0, char_offset_end=len(text))]
    elif segment_pattern:
        segments = _segment_by_pattern(text, segment_pattern)
    else:
        segments = _segment_articles(text)

    return Document.from_text(text=text, source_path=path, title=title, segments=segments, metadata=metadata)


def _extract_with_pymupdf(path: Path) -> str:
    """Extract text using pymupdf (fitz) with block-based reading order."""
    import fitz

    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        blocks = page.get_text("blocks")
        # Sort blocks by vertical position, then horizontal (reading order)
        blocks.sort(key=lambda b: (b[1], b[0]))
        page_text = "\n".join(b[4] for b in blocks if b[6] == 0)  # type 0 = text
        pages.append(page_text)
    doc.close()
    return "\n\n".join(pages)


def _extract_with_pdfplumber(path: Path) -> str:
    """Extract text using pdfplumber (better for some layouts)."""
    import pdfplumber

    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _get_page_count(path: Path) -> int:
    """Get PDF page count."""
    import fitz

    doc = fitz.open(str(path))
    count = len(doc)
    doc.close()
    return count


def _extract_title(path: Path) -> str | None:
    """Extract title from PDF metadata."""
    import fitz

    doc = fitz.open(str(path))
    metadata = doc.metadata
    doc.close()
    if metadata and metadata.get("title"):
        return metadata["title"].strip()
    return None


def _segment_articles(text: str) -> list[Segment]:
    """Heuristically segment text into articles.

    Looks for patterns common in newspaper column collections:
    - All-caps lines (likely article titles)
    - Short lines followed by blank lines and body text
    """
    # Pattern: line that's mostly uppercase, 3-80 chars, surrounded by blank lines
    title_pattern = re.compile(
        r"(?:^|\n\n)([A-Z][A-Z\s,':;\-!?.]{2,79})(?:\n\n)",
        re.MULTILINE,
    )
    matches = list(title_pattern.finditer(text))

    if len(matches) < 2:
        # Not enough titles found, return whole text as one segment
        return [Segment(index=0, title=None, text=text, char_offset_start=0, char_offset_end=len(text))]

    segments = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = match.group(1).strip()
        seg_text = text[start:end].strip()
        if len(seg_text) > 50:  # Skip tiny fragments
            segments.append(Segment(
                index=len(segments),
                title=title,
                text=seg_text,
                char_offset_start=start,
                char_offset_end=end,
            ))

    return segments if segments else [
        Segment(index=0, title=None, text=text, char_offset_start=0, char_offset_end=len(text))
    ]


def _segment_by_pattern(text: str, pattern: str) -> list[Segment]:
    """Segment text by a user-provided regex pattern."""
    compiled = re.compile(pattern, re.MULTILINE)
    matches = list(compiled.finditer(text))

    if not matches:
        return [Segment(index=0, title=None, text=text, char_offset_start=0, char_offset_end=len(text))]

    segments = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = match.group(0).strip()
        seg_text = text[start:end].strip()
        segments.append(Segment(index=i, title=title, text=seg_text, char_offset_start=start, char_offset_end=end))

    return segments

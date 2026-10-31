"""PDF file ingestion using pymupdf (primary) and pdfplumber (fallback)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from colophon.ingestion.normalize import normalize_text
from colophon.models.document import Document, DocumentBlock, Segment, text_blocks


def ingest_pdf(
    path: Path,
    *,
    no_segment: bool = False,
    segment_pattern: str | None = None,
) -> Document:
    """Ingest a PDF file, extracting and normalizing text."""
    text, blocks = _extract_layout(path)
    warnings = ["PDF blocks are physical layout regions; paragraph boundaries require review."]
    extractor = "pymupdf-layout-2"

    # If pymupdf yields very little text, try pdfplumber
    page_count = _get_page_count(path)
    chars_per_page = len(text) / max(page_count, 1)
    if chars_per_page < 100:
        pdfplumber_text = _extract_with_pdfplumber(path)
        if len(pdfplumber_text) > len(text):
            text = normalize_text(pdfplumber_text)
            blocks = _unverified_blocks(text)
            extractor = "pdfplumber-2"

    # If still very little text, suggest OCR
    chars_per_page = len(text) / max(page_count, 1)
    if chars_per_page < 50:
        try:
            from colophon.ingestion.ocr import ocr_pdf

            ocr_path = ocr_pdf(path)
            try:
                text, blocks = _extract_layout(ocr_path)
                extractor = "ocr-pymupdf-2"
            finally:
                ocr_path.unlink(missing_ok=True)
        except (ImportError, RuntimeError, OSError) as error:
            warnings.append(f"OCR unavailable or failed ({type(error).__name__}); extraction may be incomplete.")

    text = normalize_text(text)
    metadata = {
        "page_count": str(page_count),
        "extractor": extractor,
        "normalization_version": "2",
        "paragraph_quality": "unverified",
    }
    if not text.strip():
        warnings.append("No readable text was extracted.")

    # Extract title from PDF metadata if available
    title = _extract_title(path) or path.stem

    if no_segment:
        segments = [Segment(index=0, title=None, text=text, char_offset_start=0, char_offset_end=len(text))]
    elif segment_pattern:
        segments = _segment_by_pattern(text, segment_pattern)
    else:
        segments = _segment_articles(text)

    return Document.from_text(
        text=text, source_path=path, title=title, segments=segments, metadata=metadata, blocks=blocks, warnings=warnings
    )


def _unverified_blocks(text: str) -> list[DocumentBlock]:
    return [
        block.model_copy(update={"kind": "unknown", "provenance_quality": "unverified"}) for block in text_blocks(text)
    ]


def _extract_layout(path: Path) -> tuple[str, list[DocumentBlock]]:
    """Retain physical source regions without asserting they are prose paragraphs."""
    import pymupdf as fitz

    pieces: list[str] = []
    items: list[DocumentBlock] = []
    offset = 0
    with fitz.open(str(path)) as pdf:
        for page_index in range(pdf.page_count):
            page = pdf[page_index]
            regions = sorted(page.get_text("blocks"), key=lambda value: (value[1], value[0]))
            for region in regions:
                if region[6] != 0:
                    continue
                text = " ".join(normalize_text(region[4]).split())
                if not text:
                    continue
                if pieces:
                    offset += 2
                items.append(
                    DocumentBlock(
                        id=hashlib.sha256(f"{page_index}:{region[:4]}:{text}".encode()).hexdigest()[:16],
                        kind="unknown",
                        text=text,
                        attribution="author",
                        page=page_index + 1,
                    bbox=(float(region[0]), float(region[1]), float(region[2]), float(region[3])),
                        char_offset_start=offset,
                        char_offset_end=offset + len(text),
                        provenance_quality="region_only",
                    )
                )
                pieces.append(text)
                offset += len(text)
    return "\n\n".join(pieces), items


def _extract_with_pymupdf(path: Path) -> str:
    """Extract text using pymupdf (fitz) with block-based reading order."""
    return _extract_layout(path)[0]


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
    import pymupdf as fitz

    doc = fitz.open(str(path))
    count = len(doc)
    doc.close()
    return count


def _extract_title(path: Path) -> str | None:
    """Extract title from PDF metadata."""
    import pymupdf as fitz

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
            segments.append(
                Segment(
                    index=len(segments),
                    title=title,
                    text=seg_text,
                    char_offset_start=start,
                    char_offset_end=end,
                )
            )

    return (
        segments
        if segments
        else [Segment(index=0, title=None, text=text, char_offset_start=0, char_offset_end=len(text))]
    )


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

"""TXT and MD file ingestion."""

from __future__ import annotations

import re
from pathlib import Path

from colophon.ingestion.normalize import normalize_text
from colophon.models.document import Document, Segment


def ingest_text(
    path: Path,
    *,
    no_segment: bool = False,
    segment_pattern: str | None = None,
) -> Document:
    """Ingest a plain text or markdown file."""
    raw = path.read_text(encoding="utf-8")
    text = normalize_text(raw)

    if no_segment:
        segments = [Segment(index=0, title=None, text=text, char_offset_start=0, char_offset_end=len(text))]
    elif segment_pattern:
        segments = _segment_by_pattern(text, segment_pattern)
    elif path.suffix.lower() == ".md":
        segments = _segment_markdown(text)
    else:
        segments = _segment_text(text)

    return Document.from_text(text=text, source_path=path, segments=segments)


def _segment_markdown(text: str) -> list[Segment]:
    """Segment markdown by top-level headers."""
    pattern = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        return [Segment(index=0, title=None, text=text, char_offset_start=0, char_offset_end=len(text))]

    segments = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = match.group(2).strip()
        seg_text = text[start:end].strip()
        segments.append(Segment(index=i, title=title, text=seg_text, char_offset_start=start, char_offset_end=end))

    return segments


def _segment_text(text: str) -> list[Segment]:
    """Segment plain text by blank-line-separated title-like lines."""
    # Look for short lines (likely titles) followed by body text
    blocks = re.split(r"\n\n+", text)
    if len(blocks) <= 1:
        return [Segment(index=0, title=None, text=text, char_offset_start=0, char_offset_end=len(text))]

    # If all blocks are substantial, just return them as segments
    segments = []
    offset = 0
    for i, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue
        start = text.index(block, offset)
        end = start + len(block)
        segments.append(Segment(index=i, title=None, text=block, char_offset_start=start, char_offset_end=end))
        offset = end

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
        title = match.group(0).strip() if match.group(0) else None
        seg_text = text[start:end].strip()
        segments.append(Segment(index=i, title=title, text=seg_text, char_offset_start=start, char_offset_end=end))

    return segments

"""EPUB file ingestion using ebooklib + BeautifulSoup."""

from __future__ import annotations

import re
from pathlib import Path

from colophon.ingestion.normalize import normalize_text
from colophon.models.document import Document, Segment


def ingest_epub(
    path: Path,
    *,
    no_segment: bool = False,
    segment_pattern: str | None = None,
) -> Document:
    """Ingest an EPUB file, extracting chapters as segments."""
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    book = epub.read_epub(str(path))

    # Extract metadata
    title = None
    author = None
    metadata: dict[str, str] = {}

    titles = book.get_metadata("DC", "title")
    if titles:
        title = titles[0][0]
    creators = book.get_metadata("DC", "creator")
    if creators:
        author = creators[0][0]
    publishers = book.get_metadata("DC", "publisher")
    if publishers:
        metadata["publisher"] = publishers[0][0]

    # Extract text from document items in spine order
    spine_ids = [item_id for item_id, _ in book.spine]
    items_by_id = {item.get_id(): item for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)}

    chapters: list[tuple[str | None, str]] = []
    for item_id in spine_ids:
        item = items_by_id.get(item_id)
        if item is None:
            continue
        content = item.get_body_content()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        soup = BeautifulSoup(content, "html.parser")
        text = "\n".join(p.get_text() for p in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]))
        if text.strip():
            # Try to extract chapter title from first heading
            heading = soup.find(["h1", "h2", "h3"])
            chapter_title = heading.get_text().strip() if heading else None
            chapters.append((chapter_title, text.strip()))

    # Combine all text
    full_text = normalize_text("\n\n".join(text for _, text in chapters))

    if no_segment:
        segments = [Segment(index=0, title=None, text=full_text, char_offset_start=0, char_offset_end=len(full_text))]
    elif segment_pattern:
        segments = _segment_by_pattern(full_text, segment_pattern)
    else:
        # Use chapters as segments
        segments = []
        offset = 0
        for i, (ch_title, ch_text) in enumerate(chapters):
            ch_text = normalize_text(ch_text)
            # Find this chapter's text in the full text
            try:
                start = full_text.index(ch_text[:80], offset)
            except ValueError:
                start = offset
            end = start + len(ch_text)
            segments.append(Segment(
                index=i,
                title=ch_title,
                text=ch_text,
                char_offset_start=start,
                char_offset_end=end,
            ))
            offset = end

    metadata["chapter_count"] = str(len(chapters))
    return Document.from_text(
        text=full_text,
        source_path=path,
        title=title or path.stem,
        author=author,
        segments=segments,
        metadata=metadata,
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

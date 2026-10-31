"""EPUB file ingestion using ebooklib + BeautifulSoup."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

from colophon.ingestion.normalize import normalize_text
from colophon.models.document import Document, DocumentBlock, Segment

if TYPE_CHECKING:
    from bs4.element import Tag


def _logical_blocks(root: Tag) -> list[tuple[Tag, str]]:
    """Emit inline runs once, splitting at nested and sibling block boundaries."""
    from bs4.element import CData, NavigableString, Tag

    prose_tags = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote"}
    boundaries = prose_tags | {
        "address", "article", "aside", "dd", "div", "dl", "dt", "fieldset", "figcaption", "figure",
        "footer", "form", "header", "hr", "main", "nav", "ol", "pre", "section", "table", "tbody",
        "td", "tfoot", "th", "thead", "tr", "ul",
    }
    records: list[tuple[Tag, str]] = []
    buffer: list[str] = []

    def flush(owner: Tag | None) -> None:
        text = " ".join(normalize_text("".join(buffer)).split())
        buffer.clear()
        if text and owner is not None:
            records.append((owner, text))

    def walk(node: Tag, owner: Tag | None) -> None:
        boundary = node.name in boundaries
        if boundary:
            flush(owner)
        current_owner = node if node.name in prose_tags else owner
        for child in node.children:
            if isinstance(child, Tag):
                walk(child, current_owner)
            # Match get_text's visible string types; comments and script/style
            # strings do not become prose merely because they are inside a list.
            elif type(child) in (NavigableString, CData) and current_owner is not None:
                buffer.append(str(child))
        if boundary:
            flush(current_owner)

    walk(root, None)
    return records


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
    blocks: list[DocumentBlock] = []
    pieces: list[str] = []
    offset = 0
    for item_id in spine_ids:
        item = items_by_id.get(item_id)
        if item is None:
            continue
        content = item.get_body_content()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        soup = BeautifulSoup(content, "html.parser")
        chapter_pieces = []
        for node, text_part in _logical_blocks(soup):
            if pieces:
                offset += 2
            furniture = node.find_parent(["nav", "header", "footer"]) is not None
            quoted = node.name == "blockquote" or node.find_parent("blockquote") is not None
            kind = (
                "furniture"
                if furniture
                else "quotation"
                if quoted
                else "heading"
                if node.name in {"h1", "h2", "h3", "h4", "h5", "h6"}
                else "paragraph"
            )
            attribution = "quoted" if quoted else "unknown" if furniture else "author"
            blocks.append(
                DocumentBlock(
                    id=hashlib.sha256(f"{item_id}:{len(blocks)}:{text_part}".encode()).hexdigest()[:16],
                    kind=kind,
                    attribution=attribution,
                    text=text_part,
                    char_offset_start=offset,
                    char_offset_end=offset + len(text_part),
                    source_item=item_id,
                )
            )
            offset += len(text_part)
            pieces.append(text_part)
            chapter_pieces.append(text_part)
        text = "\n\n".join(chapter_pieces)
        if text.strip():
            # Try to extract chapter title from first heading
            heading = soup.find(["h1", "h2", "h3"])
            chapter_title = heading.get_text().strip() if heading else None
            chapters.append((chapter_title, text.strip()))

    # Combine all text
    full_text = "\n\n".join(pieces)

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
            segments.append(
                Segment(
                    index=i,
                    title=ch_title,
                    text=ch_text,
                    char_offset_start=start,
                    char_offset_end=end,
                )
            )
            offset = end

    metadata["chapter_count"] = str(len(chapters))
    metadata.update({"extractor": "epub-3", "normalization_version": "2", "paragraph_quality": "explicit"})
    return Document.from_text(
        text=full_text,
        source_path=path,
        title=title or path.stem,
        author=author,
        segments=segments,
        metadata=metadata,
        blocks=blocks,
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

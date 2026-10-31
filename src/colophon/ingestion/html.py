"""Local HTML prose extraction, including legacy BR-separated essay pages.

Documents are data. Script/style content is never interpreted or executed.
Normalized offsets identify the emitted text, not raw HTML byte positions.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlparse

from colophon.ingestion.normalize import normalize_text
from colophon.models.document import Document, DocumentBlock, Segment

_BLOCK_TAGS = {"p", "div", "section", "article", "main", "li", "ul", "ol", "table", "tr", "td"}
_FURNITURE = {"nav", "header", "footer", "aside", "form", "pre", "figure", "figcaption"}
_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_DATE = re.compile(
    _MONTH + r"\s+\d{4}(?:\s+\d{1,2})?(?:\s*\((?:rev\.?|revised|updated)\s+" + _MONTH + r"\s+\d{4}\))?",
    re.I,
)
Kind = Literal["paragraph", "heading", "quotation", "furniture", "unknown"]
Attribution = Literal["author", "quoted", "editor", "translator", "unknown"]
_ATTRIBUTION = re.compile(r"^(?:-{1,3}|[–—])\s+\S")
_LISP_TOKEN = re.compile(r"\(\s*([^\s()]+)")
_LISP_FORMS = {"if", "let", "let*", "cond", "lambda", "defun", "define", "setq", "setf", "progn", "when", "unless"}


def _display_code(text: str) -> bool:
    """Conservative unmarked display-Lisp detection, never inline prose matching."""
    if not text.startswith("(") or not text.endswith(")"):
        return False
    depth = 0
    for index, char in enumerate(text):
        depth += (char == "(") - (char == ")")
        if depth < 0 or (depth == 0 and index != len(text) - 1):
            return False
    tokens = _LISP_TOKEN.findall(text)
    if depth or len(tokens) < 2:
        return False
    signals = [token in _LISP_FORMS or bool(re.fullmatch(r"[+*/=<>-]+", token)) for token in tokens]
    return all(signals)


def _data_table(tag: Any) -> bool:
    """A multirow numeric matrix is display data, not a prose paragraph per cell."""
    if tag.name != "table" or len(tag.find_all("tr")) < 2 or tag.find("table") is not None:
        return False
    # Legacy HTML often omits TD end tags, so html.parser nests cells. Count
    # leaf-cell values instead of duplicating the following cell's text.
    values = [cell.get_text(" ", strip=True) for cell in tag.find_all(["td", "th"]) if cell.find(["td", "th"]) is None]
    if len(values) < 4:
        return False
    numeric = sum(bool(re.fullmatch(r"[$€£]?[+-]?\d[\d.,]*(?:\s*%)?", value)) for value in values)
    return numeric / len(values) >= 0.5


def _promotional_table(tag: Any) -> bool:
    """Recognize a small linked, colored legacy callout, not ordinary prose."""
    if tag.name != "table" or len(tag.get_text(" ", strip=True)) > 220:
        return False
    if tag.find("td", bgcolor=True) is None or tag.find("b") is None:
        return False
    for anchor in tag.find_all("a", href=True):
        host = urlparse(str(anchor.get("href", ""))).hostname or ""
        if host.casefold().removeprefix("www.") == "ycombinator.com":
            return True
    return False


def _epigraph_table(tag: Any) -> bool:
    """A leading boxed quotation has a separate dash-prefixed attribution."""
    if tag.name != "table" or tag.find("td", bgcolor=True) is None:
        return False
    text = normalize_text(tag.get_text(" ", strip=True))
    if len(text) > 2500 or not text.startswith('"') or len(tag.find_all("br")) < 2:
        return False
    lines = normalize_text(tag.get_text("\n", strip=True)).splitlines()
    return any(_ATTRIBUTION.match(line.strip()) for line in lines)


def _locator(tag: Any) -> str:
    parts = []
    while getattr(tag, "name", None) and tag.name != "[document]":
        previous = sum(1 for sibling in tag.previous_siblings if getattr(sibling, "name", None) == tag.name)
        parts.append(f"{tag.name}:nth-of-type({previous + 1})")
        tag = tag.parent
    return " > ".join(reversed(parts))


def ingest_html(path: Path, *, no_segment: bool = False, segment_pattern: str | None = None) -> Document:
    """Extract labeled logical blocks, retaining excluded furniture for inspection."""
    from bs4 import BeautifulSoup
    from bs4.element import Comment, NavigableString, Tag

    raw = path.read_bytes()
    soup = BeautifulSoup(raw, "html.parser")
    title = " ".join(soup.title.get_text().split()) if soup.title else path.stem
    removed = len(soup.find_all(["script", "style", "noscript", "template"]))
    for tag in soup.find_all(["script", "style", "noscript", "template"]):
        tag.decompose()
    root = soup.find("article") or soup.find("main")
    root_method = "semantic"
    warnings = []
    if root is None:
        candidates = soup.select('font[face="verdana"][size="2"]')
        if candidates:
            root = max(candidates, key=lambda tag: len(tag.get_text()))
            root_method = "legacy-font"
            warnings.append("Legacy font-based essay root selected; body and tail boundaries require source review.")
        else:
            root = soup.body or soup
            root_method = "body-fallback"
            warnings.append("No semantic article root found; verify extracted body and furniture boundaries.")
    root_locator = _locator(root)
    records: list[tuple[str, str, str, str]] = []
    buffer: list[str] = []
    buffer_bold: list[bool] = []
    buffer_locator = root_locator
    breaks = 0
    tail = False
    context_kind = "paragraph"
    context_attribution = "author"
    skip_closing_reference = False
    epigraph_context = False
    promotional_regions = 0
    epigraph_regions = 0
    data_table_regions = 0
    display_code_regions = 0

    def flush() -> None:
        nonlocal breaks, tail, display_code_regions
        value = " ".join(normalize_text("".join(buffer)).split())
        standalone_bold = bool(buffer_bold) and all(buffer_bold)
        buffer.clear()
        buffer_bold.clear()
        breaks = 0
        if not value:
            return
        kind, attribution = context_kind, context_attribution
        if kind == "paragraph" and value.casefold() in {
            "notes",
            "footnotes",
            "endnotes",
            "acknowledgments",
            "acknowledgements",
        }:
            tail = True
        leading = not any(item[1] == "paragraph" and item[2] == "author" for item in records)
        if kind == "paragraph" and leading and _DATE.fullmatch(value):
            kind, attribution = "furniture", "unknown"
        if kind == "paragraph" and value.startswith("(") and value.endswith(")") and leading:
            if re.search(r"\b(?:essay|article|lecture|talk)\b", value, re.I):
                kind, attribution = "furniture", "unknown"
        if (
            kind == "paragraph"
            and re.match(r"^Thanks\b", value)
            and re.search(r"\b(?:drafts?|suggestions|reading|comments)\b", value, re.I)
        ):
            tail = True
        if kind == "paragraph" and root_method == "legacy-font" and standalone_bold and len(value.split()) <= 12:
            kind = "heading"
        if epigraph_context and _ATTRIBUTION.match(value):
            kind, attribution = "furniture", "unknown"
        if kind == "paragraph" and _display_code(value):
            kind, attribution = "furniture", "unknown"
            display_code_regions += 1
        if tail:
            kind, attribution = "furniture", "unknown"
        records.append((value, kind, attribution, buffer_locator))

    def add_text(value: str, locator: str, *, bold: bool = False) -> None:
        nonlocal breaks, buffer_locator, skip_closing_reference
        if skip_closing_reference and value.strip():
            value = re.sub(r"^(\s*)\]", r"\1", value, count=1)
            skip_closing_reference = False
        if not value.strip():
            if buffer:
                buffer.append(" ")
            return
        if breaks:
            buffer.append(" ")
        breaks = 0
        if not buffer:
            buffer_locator = locator
        buffer.append(value)
        buffer_bold.append(bold)

    def walk(node: Any) -> None:
        nonlocal breaks, context_kind, context_attribution, tail, skip_closing_reference
        nonlocal epigraph_context, promotional_regions, epigraph_regions
        nonlocal data_table_regions
        if isinstance(node, Comment):
            return
        if isinstance(node, NavigableString):
            bold = any(getattr(parent, "name", None) in {"b", "strong"} for parent in node.parents)
            add_text(str(node), _locator(node.parent), bold=bold)
            return
        if not isinstance(node, Tag):
            return
        name = node.name.lower()
        if name in {"script", "style", "noscript", "template", "head", "title"}:
            return
        if name == "br":
            breaks += 1
            if breaks >= 2:
                flush()
            # html.parser can nest content inside malformed non-void BR tags.
            for child in node.children:
                walk(child)
            return
        if name == "hr":
            flush()
            return
        attrs = " ".join([str(node.get("id", "")), str(node.get("class", "")), str(node.get("role", ""))]).lower()
        is_note = bool(re.search(r"(?:footnotes?|endnotes?|doc-endnotes)", attrs))
        hidden = node.has_attr("hidden") or node.get("aria-hidden") == "true"
        if name == "a":
            href = str(node.get("href", ""))
            if re.search(r"#(?:f\d+n|fn\d+|footnote\d+)$", href, re.I):
                if buffer and buffer[-1].rstrip().endswith("["):
                    buffer[-1] = buffer[-1].rstrip()[:-1]
                    skip_closing_reference = True
                return  # Inline note number is apparatus, not an author word.
        if name == "sup" and node.find("a", href=True):
            return
        if name == "q":
            add_text('"', _locator(node))
            for child in node.children:
                walk(child)
            add_text('"', _locator(node))
            return
        boundary = (
            name in _BLOCK_TAGS
            or name in _FURNITURE
            or name == "blockquote"
            or bool(re.fullmatch("h[1-6]", name))
            or is_note
            or hidden
        )
        if boundary and node is not root:
            flush()
        previous = context_kind, context_attribution
        previous_epigraph = epigraph_context
        leading = not any(item[1] == "paragraph" and item[2] == "author" for item in records)
        promotional = (
            root_method == "legacy-font" and leading and context_attribution == "author" and _promotional_table(node)
        )
        epigraph = (
            root_method == "legacy-font"
            and leading
            and context_attribution == "author"
            and not epigraph_context
            and _epigraph_table(node)
        )
        data_table = _data_table(node)
        if promotional:
            context_kind, context_attribution = "furniture", "unknown"
            promotional_regions += 1
        elif epigraph:
            context_kind, context_attribution = "quotation", "quoted"
            epigraph_context = True
            epigraph_regions += 1
        elif data_table:
            context_kind, context_attribution = "furniture", "unknown"
            data_table_regions += 1
        elif name in _FURNITURE or is_note or hidden:
            context_kind, context_attribution = "furniture", "unknown"
        elif name == "blockquote":
            context_kind, context_attribution = "quotation", "quoted"
        elif re.fullmatch("h[1-6]", name):
            context_kind = "heading"
            if context_attribution == "author" and node.get_text(" ", strip=True).casefold() in {
                "notes",
                "footnotes",
                "endnotes",
            }:
                tail = True
        for child in node.children:
            walk(child)
        if boundary and node is not root:
            flush()
        context_kind, context_attribution = previous
        epigraph_context = previous_epigraph

    walk(root)
    flush()
    for index, (value, kind, attribution, locator) in enumerate(records):
        if kind == "paragraph" and value.casefold() == "or" and 0 < index < len(records) - 1:
            if _display_code(records[index - 1][0]) and _display_code(records[index + 1][0]):
                records[index] = (value, "furniture", "unknown", locator)
    if not records:
        raise ValueError(f"No readable body text in HTML: {path.name}")
    pieces = []
    blocks = []
    offset = 0
    for index, (value, kind, attribution, locator) in enumerate(records):
        if pieces:
            offset += 2
        blocks.append(
            DocumentBlock(
                id=hashlib.sha256(f"{index}:{locator}:{value}".encode()).hexdigest()[:16],
                text=value,
                kind=cast(Kind, kind),
                attribution=cast(Attribution, attribution),
                char_offset_start=offset,
                char_offset_end=offset + len(value),
                source_item=locator,
                provenance_quality="region_only",
            )
        )
        pieces.append(value)
        offset += len(value)
    text = "\n\n".join(pieces)
    if segment_pattern and not no_segment:
        from colophon.ingestion.text import _segment_by_pattern

        segments = _segment_by_pattern(text, segment_pattern)
    else:
        segments = [Segment(index=0, title=title, text=text, char_offset_start=0, char_offset_end=len(text))]
    return Document.from_text(
        text=text,
        source_path=path,
        title=title,
        blocks=blocks,
        segments=segments,
        warnings=warnings,
        metadata={
            "extractor": "html-2",
            "normalization_version": "2",
            "paragraph_quality": "explicit",
            "root_method": root_method,
            "root_locator": root_locator,
            "raw_source_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_source_path": str(path.resolve()),
            "removed_active_elements": str(removed),
            "promotional_regions": str(promotional_regions),
            "epigraph_regions": str(epigraph_regions),
            "data_table_regions": str(data_table_regions),
            "display_code_regions": str(display_code_regions),
            "offset_space": "normalized document text; DOM locators are source-region references",
        },
    )

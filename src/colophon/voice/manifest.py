"""Load local corpora, preserve ownership and reject leaking evaluation splits."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from colophon.ingestion import ingest
from colophon.models.document import Document, DocumentBlock, text_blocks
from colophon.voice.models import CorpusEntry, CorpusManifest


@dataclass(frozen=True)
class LoadedDocument:
    entry: CorpusEntry
    document: Document
    source_hash: str
    normalized_hash: str
    author_document: Document
    excluded_word_count: int
    warnings: list[str]


def quoted_ranges(text: str) -> tuple[list[tuple[int, int]], list[str]]:
    """Conservative paired double/guillemet quotes; leave apostrophes as words."""
    spans: list[tuple[int, int]] = []
    warnings = []
    for opening, closing in [('"', '"'), ("«", "»"), ("「", "」"), ("『", "』")]:
        start = None
        for index, char in enumerate(text):
            if start is None and char == opening:
                start = index
            elif start is not None and char == closing:
                spans.append((start, index + 1))
                start = None
        if start is not None:
            spans.append((start, len(text)))
            warnings.append("Unclosed quotation excluded through end of document; review source boundaries.")
    return spans, warnings


def _source_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Corpus sources must stay inside the manifest directory")
    if any(part.startswith(".") for part in Path(value).parts if part not in {".", ".."}):
        raise ValueError("Hidden files are not corpus sources")
    if path.suffix.lower() not in {".txt", ".md", ".pdf", ".epub", ".html", ".htm"} and not path.name.endswith(
        ".colophon.json"
    ):
        raise ValueError("Use TXT, MD, PDF, EPUB, HTML or .colophon.json corpus sources")
    if not path.is_file():
        raise ValueError(f"Corpus source does not exist: {value}")
    return path


def _author_view(document: Document, entry: CorpusEntry) -> tuple[Document, int, list[str]]:
    warnings = list(document.warnings)
    if not document.blocks:
        document = document.model_copy(update={"blocks": text_blocks(document.text)})
        warnings.append("Legacy document has inferred plain-text blocks; original parser provenance is unavailable.")
    spans = list(
        (block.char_offset_start, block.char_offset_end)
        for block in document.blocks
        if block.kind in {"heading", "furniture", "quotation"} or block.attribution != "author"
    )
    for excluded in entry.excluded_spans:
        if excluded.end > len(document.text):
            raise ValueError(f"Excluded span exceeds normalized text in {entry.id!r}")
        spans.append((excluded.start, excluded.end))
    chars = list(document.text)
    for start, end in spans:
        for index in range(start, end):
            if not chars[index].isspace():
                chars[index] = " "
    if entry.quotation_policy == "exclude_all":
        # Excluded code, headings and notes cannot open a quotation that masks
        # subsequent author prose. Pair quotes only after those regions are removed.
        quote_spans, quote_warnings = quoted_ranges("".join(chars))
        warnings.extend(quote_warnings)
        for start, end in quote_spans:
            for index in range(start, end):
                if not chars[index].isspace():
                    chars[index] = " "
    else:
        warnings.append("Only explicitly attributed quotation blocks/spans excluded; inline quotations may remain.")
    masked = "".join(chars)
    # Removed quotations must not split a paragraph or shift source evidence offsets.
    blocks: list[DocumentBlock] = []
    for block in document.blocks:
        value = masked[block.char_offset_start : block.char_offset_end]
        if not value.strip():
            continue
        blocks.append(block.model_copy(update={"text": value}))
    metadata = {**document.metadata, "normalization_version": "2", "quotation_policy": entry.quotation_policy}
    if entry.paragraphs_verified:
        metadata["paragraph_quality"] = "reviewed"
        blocks = [
            block.model_copy(update={"kind": "paragraph"}) if block.kind == "unknown" else block for block in blocks
        ]
    elif metadata.get("paragraph_quality") == "unverified":
        warnings.append("Paragraph statistics unavailable until PDF paragraph boundaries are reviewed.")
    metadata["source_document_id"] = document.id
    metadata["source_hash"] = hashlib.sha256(document.text.encode()).hexdigest()
    view = Document.from_text(
        text=masked,
        source_path=document.source_path,
        title=document.title,
        author=entry.effective_author,
        blocks=blocks,
        metadata=metadata,
        warnings=warnings,
    )
    # Count excluded lexical words without claiming a tokenizer-specific author total.
    excluded_words = max(0, len(re.findall(r"\w+", document.text)) - len(re.findall(r"\w+", masked)))
    return view, excluded_words, warnings


def _shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[i : i + 5]) for i in range(max(0, len(words) - 4))}


def validate_loaded(loaded: list[LoadedDocument]) -> None:
    """Check full sources and included author prose for duplicates and split leakage."""
    if not loaded:
        raise ValueError("Corpus has no included documents")
    ids: set[str] = set()
    content_ids: dict[str, str] = {}
    author_content_ids: dict[str, str] = {}
    works: dict[str, str] = {}
    shingles = []
    author_shingles = []
    for item in loaded:
        entry = item.entry
        if entry.id in ids:
            raise ValueError(f"Duplicate document ID: {entry.id}")
        ids.add(entry.id)
        if entry.work_id in works and works[entry.work_id] != entry.split:
            raise ValueError(f"Work {entry.work_id!r} crosses splits")
        works[entry.work_id] = entry.split
        canonical = " ".join(item.document.text.casefold().split())
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        if digest in content_ids:
            raise ValueError(f"Duplicate text in {entry.id!r} and {content_ids[digest]!r}")
        content_ids[digest] = entry.id
        current = _shingles(canonical)
        for previous, other in shingles:
            denominator = min(len(current), len(other))
            if entry.split != previous.entry.split and denominator >= 20:
                overlap = len(current & other) / denominator
                if overlap >= 0.85:
                    raise ValueError(f"Near-duplicate text crosses splits: {entry.id!r}, {previous.entry.id!r}")
        shingles.append((item, current))
        # Analysis consumes the masked author view. Compare its canonical prose
        # as well, so excluded material cannot hide leakage or affect identity.
        # Canonicalization is comparison-only: evidence keeps its source offsets.
        author_canonical = " ".join(item.author_document.text.casefold().split())
        author_digest = hashlib.sha256(author_canonical.encode()).hexdigest()
        if author_digest in author_content_ids:
            raise ValueError(f"Duplicate author text in {entry.id!r} and {author_content_ids[author_digest]!r}")
        author_content_ids[author_digest] = entry.id
        author_current = _shingles(author_canonical)
        for previous, other in author_shingles:
            denominator = min(len(author_current), len(other))
            if entry.split != previous.entry.split and denominator >= 20:
                overlap = len(author_current & other) / denominator
                if overlap >= 0.85:
                    raise ValueError(f"Near-duplicate author text crosses splits: {entry.id!r}, {previous.entry.id!r}")
        author_shingles.append((item, author_current))


def load_corpus(path: Path) -> list[LoadedDocument]:
    """Load a JSON manifest and contained sources; never fetch documents or models."""
    path = path.resolve()
    manifest = CorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))
    loaded = []
    for entry in manifest.documents:
        if not entry.include:
            continue
        translators = {person.name for person in entry.contributors if person.role == "translator"}
        if translators and entry.effective_author not in translators:
            raise ValueError(f"{entry.id!r}: assign voice_author to the translator or exclude this translation")
        source = _source_path(path.parent, entry.path)
        raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        document = (
            Document.model_validate_json(source.read_text())
            if source.name.endswith(".colophon.json")
            else ingest(source)
        )
        if not document.text.strip():
            raise ValueError(f"{entry.id!r} has no extracted text")
        if not document.blocks:
            document = document.model_copy(update={"blocks": text_blocks(document.text)})
        view, excluded_count, warnings = _author_view(document, entry)
        if any(person.role == "editor" for person in entry.contributors):
            warnings.append("Editor contribution is recorded; unmarked editorial prose remains a possible confounder.")
        if not view.text.strip():
            raise ValueError(f"{entry.id!r} has no author prose after exclusions")
        loaded.append(
            LoadedDocument(
                entry=entry,
                document=document,
                source_hash=raw_hash,
                normalized_hash=hashlib.sha256(document.text.encode()).hexdigest(),
                author_document=view,
                excluded_word_count=excluded_count,
                warnings=warnings,
            )
        )
    validate_loaded(loaded)
    return loaded

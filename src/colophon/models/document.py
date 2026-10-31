"""Document models for ingested text."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class DocumentBlock(BaseModel):
    """Logical text with offsets in normalized document text, and source regions."""

    model_config = ConfigDict(frozen=True)
    id: str
    kind: Literal["paragraph", "heading", "quotation", "furniture", "unknown"] = "paragraph"
    text: str
    char_offset_start: int
    char_offset_end: int
    attribution: Literal["author", "quoted", "editor", "translator", "unknown"] = "author"
    page: int | None = None
    source_item: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    provenance_quality: Literal["normalized_offsets", "region_only", "unverified"] = "normalized_offsets"


def text_blocks(text: str, *, markdown: bool = False) -> list[DocumentBlock]:
    """Keep every nonempty paragraph, including one-word beats, with exact offsets."""
    blocks = []
    cursor = 0
    boundaries = [(m.start(), m.end()) for m in re.finditer(r"\n[ \t]*\n", text)]
    for boundary, next_cursor in [*boundaries, (len(text), len(text))]:
        raw = text[cursor:boundary]
        start = cursor + len(raw) - len(raw.lstrip())
        end = cursor + len(raw.rstrip())
        cursor = next_cursor
        if end <= start:
            continue
        value = text[start:end]
        heading = markdown and value.startswith("#") and "\n" not in value
        kind = "heading" if heading else "paragraph"
        blocks.append(
            DocumentBlock(
                id=hashlib.sha256(f"{start}:{end}:{value}".encode()).hexdigest()[:16],
                text=value,
                kind=kind,
                char_offset_start=start,
                char_offset_end=end,
            )
        )
    return blocks


class Segment(BaseModel):
    """A segment of a document (e.g., an article, chapter)."""

    model_config = ConfigDict(frozen=True)

    index: int
    title: str | None = None
    text: str
    char_offset_start: int
    char_offset_end: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def word_count(self) -> int:
        return len(self.text.split())


class Document(BaseModel):
    """A normalized, ingested document ready for analysis."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="SHA-256 hash of the text content.")
    source_path: str
    title: str
    author: str | None = None
    text: str
    segments: list[Segment] = Field(default_factory=list)
    blocks: list[DocumentBlock] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_blocks(self):
        previous_end = 0
        for block in self.blocks:
            start, end = block.char_offset_start, block.char_offset_end
            if start < previous_end or end <= start or self.text[start:end] != block.text:
                raise ValueError("Document blocks must be ordered, nonoverlapping exact text slices")
            previous_end = end
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @classmethod
    def from_text(
        cls,
        text: str,
        source_path: Path | str,
        title: str | None = None,
        author: str | None = None,
        segments: list[Segment] | None = None,
        metadata: dict[str, str] | None = None,
        blocks: list[DocumentBlock] | None = None,
        warnings: list[str] | None = None,
    ) -> Document:
        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        return cls(
            id=doc_id,
            source_path=str(source_path),
            title=title or Path(source_path).stem,
            author=author,
            text=text,
            segments=segments or [],
            blocks=text_blocks(text, markdown=Path(source_path).suffix.lower() == ".md") if blocks is None else blocks,
            metadata=metadata or {},
            warnings=warnings or [],
        )


class Corpus(BaseModel):
    """A collection of documents, typically by one or more authors."""

    name: str
    documents: list[Document] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

"""Document models for ingested text."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field


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
    metadata: dict[str, str] = Field(default_factory=dict)

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
    ) -> Document:
        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        return cls(
            id=doc_id,
            source_path=str(source_path),
            title=title or Path(source_path).stem,
            author=author,
            text=text,
            segments=segments or [],
            metadata=metadata or {},
        )


class Corpus(BaseModel):
    """A collection of documents, typically by one or more authors."""

    name: str
    documents: list[Document] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)

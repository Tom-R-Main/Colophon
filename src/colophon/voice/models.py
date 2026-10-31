"""Portable corpus, evidence and voice contracts. No provider or database required."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from colophon.models.features import StyleProfile


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Contributor(Record):
    name: str = Field(min_length=1)
    role: Literal["original_author", "translator", "editor", "quoted"]


class ExcludedSpan(Record):
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("Excluded span end must be after start")
        return self


class CorpusEntry(Record):
    id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    author: str = Field(min_length=1)
    voice_author: str | None = None
    work_id: str = Field(min_length=1)
    language: str = "en"
    genre: str = "unspecified"
    topic: str = "unspecified"
    date: str | None = None
    split: Literal["train", "dev", "test"] = "train"
    include: bool = True
    exclusion_reason: str | None = None
    contributors: list[Contributor] = Field(default_factory=list)
    excluded_spans: list[ExcludedSpan] = Field(default_factory=list)
    paragraphs_verified: bool = False
    quotation_policy: Literal["exclude_all", "explicit_only"] = "exclude_all"

    @property
    def effective_author(self) -> str:
        return self.voice_author or self.author


class CorpusManifest(Record):
    schema_version: Literal["1"] = "1"
    name: str
    documents: list[CorpusEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids_and_work_splits(self):
        ids = [doc.id for doc in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Corpus document IDs must be unique")
        splits: dict[str, str] = {}
        for doc in self.documents:
            if not doc.include:
                continue
            if doc.work_id in splits and splits[doc.work_id] != doc.split:
                raise ValueError(f"Work {doc.work_id!r} crosses train/dev/test splits")
            splits[doc.work_id] = doc.split
        return self


class EvidenceSpan(Record):
    document_id: str
    source_hash: str
    block_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    role: Literal["support", "counterexample", "example"] = "support"

    @model_validator(mode="after")
    def exact_length(self):
        if self.end < self.start or self.end - self.start != len(self.text):
            raise ValueError("Evidence offsets must exactly cover the supplied text")
        return self


class VoiceTrait(Record):
    id: str
    category: Literal["openings", "endings", "argument", "stance", "rhythm", "transitions", "quotations"]
    statement: str
    method: Literal["rule", "annotation"] = "rule"
    status: Literal["observed", "tentative", "accepted", "rejected", "writer_supplied"] = "tentative"
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    counterexamples: list[EvidenceSpan] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    original_statement: str | None = None


class TraitCorrection(Record):
    trait_id: str
    action: Literal["accept", "reject", "replace"]
    statement: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def replacement_required(self):
        if self.action == "replace" and not (self.statement and self.statement.strip()):
            raise ValueError("Replacing a trait requires a writer-supplied statement")
        return self


class NumericSummary(Record):
    mean: float
    median: float
    minimum: float
    maximum: float
    stdev: float
    documents: int
    unit: str


class VoiceDocument(Record):
    entry: CorpusEntry
    source_path: str | None = None
    extraction_metadata: dict[str, str] = Field(default_factory=dict)
    source_hash: str
    normalized_hash: str
    measurement: StyleProfile
    author_word_count: int
    excluded_word_count: int


class VoiceProfile(Record):
    schema_version: Literal["1"] = "1"
    extractor_version: str = "1"
    profile_id: str
    author: str
    language: str
    corpus_hash: str
    config_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    documents: list[VoiceDocument]
    numerical: dict[str, NumericSummary] = Field(default_factory=dict)
    traits: list[VoiceTrait] = Field(default_factory=list)
    exemplars: list[EvidenceSpan] = Field(default_factory=list)
    corrections: list[TraitCorrection] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

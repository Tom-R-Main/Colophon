"""SQLAlchemy models for pgvector style profile storage."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class StyleProfileRow(Base):
    __tablename__ = "style_profiles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    document_title: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str | None] = mapped_column(Text)
    lang: Mapped[str | None] = mapped_column(String(10), default="en")
    word_count: Mapped[int | None] = mapped_column(Integer)
    classical_features: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    classical_vector: Mapped[list[float] | None] = mapped_column(Vector(128))
    style_embedding: Mapped[list[float] | None] = mapped_column(Vector(768))
    hybrid_vector: Mapped[list[float] | None] = mapped_column(Vector(128))
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=datetime.now)


class AuthorProfileRow(Base):
    __tablename__ = "author_profiles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    author: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    doc_count: Mapped[int | None] = mapped_column(Integer, default=0)
    mean_classical_vector: Mapped[list[float] | None] = mapped_column(Vector(128))
    mean_classical_features: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    mean_style_embedding: Mapped[list[float] | None] = mapped_column(Vector(768))
    mean_hybrid_vector: Mapped[list[float] | None] = mapped_column(Vector(128))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=datetime.now)

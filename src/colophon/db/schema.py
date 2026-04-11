"""SQLAlchemy models for pgvector style profile storage."""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class StyleProfileRow(Base):
    __tablename__ = "style_profiles"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    document_id = Column(Text, nullable=False, unique=True)
    document_title = Column(Text, nullable=False)
    author = Column(Text)
    lang = Column(String(10), default="en")
    word_count = Column(Integer)
    classical_features = Column(JSONB, nullable=False)
    classical_vector = Column(Vector(128))
    style_embedding = Column(Vector(768))
    hybrid_vector = Column(Vector(128))
    created_at = Column(DateTime(timezone=True), default=datetime.now)


class AuthorProfileRow(Base):
    __tablename__ = "author_profiles"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    author = Column(Text, unique=True, nullable=False)
    doc_count = Column(Integer, default=0)
    mean_classical_vector = Column(Vector(128))
    mean_classical_features = Column(JSONB)
    mean_style_embedding = Column(Vector(768))
    mean_hybrid_vector = Column(Vector(128))
    updated_at = Column(DateTime(timezone=True), default=datetime.now)

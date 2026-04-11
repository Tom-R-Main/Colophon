"""Database layer for pgvector style profile storage."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


def get_session(db_url: str) -> Session:
    """Create a database session from a connection URL."""
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()

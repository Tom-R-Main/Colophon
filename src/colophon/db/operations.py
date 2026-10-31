"""Transactional indexing and version/language-isolated style search."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from colophon.db.schema import AuthorProfileRow, StyleProfileRow
from colophon.embeddings.vectorize import AVAILABILITY_ORDER, VECTOR_DIM, VECTOR_SCHEMA, style_profile_to_vector
from colophon.lang import get_profile
from colophon.models.features import StyleProfile


def _validate_query(vector: list[float], lang: str, vector_schema: str, limit: int) -> None:
    get_profile(lang)
    if vector_schema != VECTOR_SCHEMA:
        raise ValueError(f"Unsupported vector schema: {vector_schema}; expected {VECTOR_SCHEMA}")
    if limit < 1:
        raise ValueError("Search limit must be positive")
    if len(vector) != VECTOR_DIM or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"Search requires {VECTOR_DIM} finite vector coordinates")
    if not any(vector):
        raise ValueError("Cannot search with an empty style vector")
    if not any(vector[88:100]):
        raise ValueError("The query has no current measurement-availability mask; re-analyze its source")


def _coverage_filters(metadata: Any, vector: list[float]) -> list[Any]:
    """Only compare coordinates that have the same measured/unmeasured status."""
    filters = []
    for index, name in enumerate(AVAILABILITY_ORDER, start=88):
        state = func.coalesce(metadata["availability"][name].astext, "not_requested")
        filters.append(state == "measured" if vector[index] > 0 else state != "measured")
    return filters


def index_profile(
    session: Session,
    profile: StyleProfile,
    classical_vector: list[float],
    *,
    author: str | None = None,
    lang: str = "en",
    style_embedding: list[float] | None = None,
) -> StyleProfileRow:
    """Commit a profile and all affected author aggregates as one transaction.

    Unversioned profiles must be re-analyzed from source. The supplied vector
    must match the current profile's coordinates, not an old cached embedding.
    Passing no author preserves an existing assignment.
    """
    if profile.language != lang:
        raise ValueError("Profile language differs from index language; re-analyze or correct --lang")
    expected = style_profile_to_vector(profile)
    if len(classical_vector) != VECTOR_DIM or not all(
        math.isfinite(value) and math.isclose(value, target, rel_tol=1e-7, abs_tol=1e-9)
        for value, target in zip(classical_vector, expected)
    ):
        raise ValueError("Supplied vector does not match the profile's current measurement identity")
    if author is not None and not author.strip():
        raise ValueError("Author cannot be blank")
    try:
        existing = session.query(StyleProfileRow).filter_by(document_id=profile.document_id).first()
        old_author = existing.author if existing is not None else None
        row = existing
        if row is None:
            row = StyleProfileRow(document_id=profile.document_id)
            session.add(row)
        row.document_title = profile.document_title
        row.classical_features = profile.model_dump(mode="json")
        row.classical_vector = classical_vector
        row.author = author if author is not None else old_author
        row.lang = lang
        row.style_embedding = style_embedding
        row.hybrid_vector = None
        row.word_count = profile.word_count
        session.flush()
        for affected_author in sorted({name for name in (old_author, row.author) if name}):
            aggregate_author(session, affected_author, commit=False)
        session.commit()
        return row
    except Exception:
        session.rollback()
        raise


def search_similar(
    session: Session,
    vector: list[float],
    *,
    vector_type: str = "classical",
    limit: int = 10,
    lang: str = "en",
    vector_schema: str = VECTOR_SCHEMA,
) -> list[dict]:
    """Search only rows with the declared current measurement contract.

    Legacy rows remain in storage but cannot enter corrected-vector results.
    Neural/hybrid vectors need their own model identity contract before reuse.
    """
    if vector_type != "classical":
        raise ValueError("Only classical search has a versioned measurement contract")
    _validate_query(vector, lang, vector_schema, limit)
    col = StyleProfileRow.classical_vector
    results = (
        session.query(
            StyleProfileRow.document_title, StyleProfileRow.author, StyleProfileRow.word_count,
            (1 - col.cosine_distance(vector)).label("similarity"),
        )
        .filter(
            col.isnot(None), StyleProfileRow.lang == lang,
            StyleProfileRow.classical_features["language"].astext == lang,
            StyleProfileRow.classical_features["vector_schema"].astext == vector_schema,
            StyleProfileRow.classical_features["schema_version"].astext == "2",
            StyleProfileRow.classical_features["analyzer_version"].astext == "2",
            *_coverage_filters(StyleProfileRow.classical_features, vector),
        )
        .order_by(col.cosine_distance(vector))
        .limit(limit)
        .all()
    )
    return [
        {"title": row.document_title, "author": row.author, "word_count": row.word_count,
         "similarity": round(float(row.similarity), 4)}
        for row in results
    ]


def search_authors(
    session: Session,
    vector: list[float],
    *,
    limit: int = 10,
    lang: str = "en",
    vector_schema: str = VECTOR_SCHEMA,
) -> list[dict]:
    """Search compatible, available author aggregates; never mix legacy means."""
    _validate_query(vector, lang, vector_schema, limit)
    col = AuthorProfileRow.mean_classical_vector
    metadata = AuthorProfileRow.mean_classical_features
    results = (
        session.query(
            AuthorProfileRow.author, AuthorProfileRow.doc_count,
            (1 - col.cosine_distance(vector)).label("similarity"),
        )
        .filter(
            col.isnot(None), metadata["language"].astext == lang,
            metadata["vector_schema"].astext == vector_schema,
            metadata["schema_version"].astext == "2",
            metadata["status"].astext == "measured",
            *_coverage_filters(metadata, vector),
        )
        .order_by(col.cosine_distance(vector))
        .limit(limit)
        .all()
    )
    return [
        {"author": row.author, "doc_count": row.doc_count, "similarity": round(float(row.similarity), 4)}
        for row in results
    ]


def aggregate_author(session: Session, author_name: str, *, commit: bool = True) -> None:
    """Recompute one compatible mean or explicitly mark the mean unavailable.

    With ``commit=False`` the caller owns commit/rollback, including deletions.
    No-document groups remove the stale author row. Incompatible documents are
    never silently omitted from an otherwise available aggregate.
    """
    try:
        session.flush()
        docs = session.query(StyleProfileRow).filter_by(author=author_name).all()
        existing = session.query(AuthorProfileRow).filter_by(author=author_name).first()
        if not docs:
            if existing is not None:
                session.delete(existing)
            if commit:
                session.commit()
            return
        languages = {doc.lang for doc in docs}
        metadata: dict[str, Any] = {
            "schema_version": "2", "vector_schema": VECTOR_SCHEMA,
            "language": next(iter(languages)) if len(languages) == 1 else None,
            "status": "unavailable", "doc_count": len(docs),
            "analyzer_version": "2",
            "availability": docs[0].classical_features.get("availability", {}),
        }
        vectors = []
        reasons = []
        coverage = set()
        if len(languages) != 1:
            reasons.append("Author documents have mixed languages")
        for doc in docs:
            try:
                profile = StyleProfile.model_validate(doc.classical_features)
                expected = style_profile_to_vector(profile)
                if profile.language != doc.lang:
                    raise ValueError("Stored profile language differs from document language")
                stored = doc.classical_vector
                if stored is None or len(stored) != VECTOR_DIM or not all(
                    math.isfinite(float(value)) and math.isclose(float(value), target, rel_tol=1e-6, abs_tol=1e-8)
                    for value, target in zip(stored, expected)
                ):
                    raise ValueError("Stored vector differs from current profile identity")
                vectors.append([float(value) for value in stored])
                coverage.add(tuple(sorted(profile.availability.items())))
            except (ValueError, TypeError) as error:
                reasons.append(f"Document {doc.document_id}: {error}")
        if len(coverage) > 1:
            reasons.append("Author documents have different measurement availability")
        mean = None
        if not reasons:
            mean = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(VECTOR_DIM)]
            metadata["status"] = "measured"
        else:
            metadata["reasons"] = reasons
        row = existing
        if row is None:
            row = AuthorProfileRow(author=author_name)
            session.add(row)
        row.doc_count = len(docs)
        row.mean_classical_vector = mean
        row.mean_classical_features = metadata
        row.updated_at = datetime.now(timezone.utc)
        session.flush()
        if commit:
            session.commit()
    except Exception:
        if commit:
            session.rollback()
        raise

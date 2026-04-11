"""Database operations for indexing and searching style profiles."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from colophon.db.schema import AuthorProfileRow, StyleProfileRow
from colophon.models.features import StyleProfile


def index_profile(
    session: Session,
    profile: StyleProfile,
    classical_vector: list[float],
    *,
    author: str | None = None,
    lang: str = "en",
    style_embedding: list[float] | None = None,
) -> StyleProfileRow:
    """Insert or update a document's style profile."""
    existing = session.query(StyleProfileRow).filter_by(document_id=profile.document_id).first()

    if existing:
        existing.classical_features = profile.model_dump(mode="json")
        existing.classical_vector = classical_vector
        existing.author = author or existing.author
        existing.style_embedding = style_embedding
        existing.word_count = profile.word_count
        session.commit()
        return existing

    row = StyleProfileRow(
        document_id=profile.document_id,
        document_title=profile.document_title,
        author=author,
        lang=lang,
        word_count=profile.word_count,
        classical_features=profile.model_dump(mode="json"),
        classical_vector=classical_vector,
        style_embedding=style_embedding,
    )
    session.add(row)
    session.commit()

    # Update author aggregate if author is set
    if author:
        aggregate_author(session, author)

    return row


def search_similar(
    session: Session,
    vector: list[float],
    *,
    vector_type: str = "classical",
    limit: int = 10,
) -> list[dict]:
    """Find nearest neighbors by style vector."""
    if vector_type == "classical":
        col = StyleProfileRow.classical_vector
    elif vector_type == "style":
        col = StyleProfileRow.style_embedding
    elif vector_type == "hybrid":
        col = StyleProfileRow.hybrid_vector
    else:
        raise ValueError(f"Unknown vector type: {vector_type}")

    results = (
        session.query(
            StyleProfileRow.document_title,
            StyleProfileRow.author,
            StyleProfileRow.word_count,
            (1 - col.cosine_distance(vector)).label("similarity"),
        )
        .filter(col.isnot(None))
        .order_by(col.cosine_distance(vector))
        .limit(limit)
        .all()
    )

    return [
        {
            "title": r.document_title,
            "author": r.author,
            "word_count": r.word_count,
            "similarity": round(float(r.similarity), 4),
        }
        for r in results
    ]


def search_authors(
    session: Session,
    vector: list[float],
    *,
    limit: int = 10,
) -> list[dict]:
    """Find nearest authors by mean classical vector."""
    results = (
        session.query(
            AuthorProfileRow.author,
            AuthorProfileRow.doc_count,
            (1 - AuthorProfileRow.mean_classical_vector.cosine_distance(vector)).label("similarity"),
        )
        .filter(AuthorProfileRow.mean_classical_vector.isnot(None))
        .order_by(AuthorProfileRow.mean_classical_vector.cosine_distance(vector))
        .limit(limit)
        .all()
    )

    return [
        {
            "author": r.author,
            "doc_count": r.doc_count,
            "similarity": round(float(r.similarity), 4),
        }
        for r in results
    ]


def aggregate_author(session: Session, author_name: str) -> None:
    """Recompute author-level mean vectors from all their documents."""
    docs = session.query(StyleProfileRow).filter_by(author=author_name).all()
    if not docs:
        return

    # Mean classical vector
    vectors = [d.classical_vector for d in docs if d.classical_vector is not None]
    if vectors:
        n = len(vectors)
        mean_vec = [sum(v[i] for v in vectors) / n for i in range(len(vectors[0]))]
    else:
        mean_vec = None

    existing = session.query(AuthorProfileRow).filter_by(author=author_name).first()
    if existing:
        existing.doc_count = len(docs)
        existing.mean_classical_vector = mean_vec
        existing.updated_at = datetime.now()
    else:
        row = AuthorProfileRow(
            author=author_name,
            doc_count=len(docs),
            mean_classical_vector=mean_vec,
        )
        session.add(row)

    session.commit()

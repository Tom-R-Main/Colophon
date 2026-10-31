"""SQLite ORM lifecycle integration and PostgreSQL query-compilation guards.

SQLite adapts DDL types for test storage only. No pgvector distance execution or
live PostgreSQL behavior is claimed by these tests.
"""

from __future__ import annotations

import pytest
from sqlalchemy import BigInteger, create_engine, event
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Query, Session

from colophon.db import operations
from colophon.db.operations import aggregate_author, index_profile, search_authors, search_similar
from colophon.db.schema import AuthorProfileRow, Base, StyleProfileRow
from colophon.embeddings.vectorize import VECTOR_SCHEMA, style_profile_to_vector
from colophon.lang import get_profile
from colophon.models.features import FunctionWordFeatures, StyleProfile


@compiles(JSONB, "sqlite")
def _sqlite_json(type_, compiler, **kwargs):
    return "JSON"


@compiles(BigInteger, "sqlite")
def _sqlite_big_integer(type_, compiler, **kwargs):
    return "INTEGER"


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def profile(doc_id="one", *, lang="en", title="A title", word="the"):
    frequencies = dict.fromkeys(get_profile(lang).function_words, 0.0)
    frequencies[word] = 100
    return StyleProfile(
        document_id=doc_id, document_title=title, language=lang,
        schema_version="2", analyzer_version="2", vector_schema=VECTOR_SCHEMA,
        word_count=10, function_words=FunctionWordFeatures(frequencies=frequencies),
        availability={"function_words": "measured"},
    )


def index(session, value, author="A"):
    return index_profile(session, value, style_profile_to_vector(value), author=author, lang=value.language)


def test_insert_update_and_aggregate_commit_together(session):
    commits = []
    event.listen(session, "after_commit", lambda session: commits.append(True))
    value = profile()
    first = index(session, value)
    original_id = first.id
    assert len(commits) == 1
    author = session.query(AuthorProfileRow).one()
    assert author.doc_count == 1
    assert author.mean_classical_features["vector_schema"] == VECTOR_SCHEMA
    assert author.mean_classical_features["language"] == "en"
    assert author.mean_classical_features["status"] == "measured"
    changed = profile(title="Updated title", word="and")
    updated = index(session, changed)
    assert len(commits) == 2
    assert updated.id == original_id
    assert updated.document_title == "Updated title"
    assert updated.classical_vector == pytest.approx(style_profile_to_vector(changed))
    assert author.mean_classical_vector == pytest.approx(updated.classical_vector)
    assert author.doc_count == 1


def test_reassignment_updates_old_and_new_author(session):
    first = index(session, profile("one"))
    index(session, profile("two", word="and"))
    index(session, profile("one", word="but"), author="B")
    assert first.author == "B"
    authors = {row.author: row for row in session.query(AuthorProfileRow).all()}
    assert set(authors) == {"A", "B"}
    assert authors["A"].doc_count == authors["B"].doc_count == 1
    remaining = session.query(StyleProfileRow).filter_by(document_id="two").one()
    assert authors["A"].mean_classical_vector == pytest.approx(remaining.classical_vector)
    index(session, profile("two", word="and"), author="B")
    assert session.query(AuthorProfileRow).filter_by(author="A").first() is None
    assert session.query(AuthorProfileRow).filter_by(author="B").one().doc_count == 2


def test_last_document_deletes_stale_author_in_callers_transaction(session):
    row = index(session, profile())
    session.delete(row)
    session.flush()
    aggregate_author(session, "A", commit=False)
    session.commit()
    assert session.query(StyleProfileRow).count() == 0
    assert session.query(AuthorProfileRow).count() == 0


def test_aggregate_failure_rolls_back_profile_and_all_author_changes(session, monkeypatch):
    index(session, profile())

    original_aggregate = operations.aggregate_author

    def failing_aggregate(session, author, *, commit=True):
        if author == "B":
            raise RuntimeError("forced aggregate failure")
        return original_aggregate(session, author, commit=commit)

    monkeypatch.setattr(operations, "aggregate_author", failing_aggregate)
    with pytest.raises(RuntimeError, match="forced aggregate failure"):
        index(session, profile(title="Must roll back", word="and"), author="B")
    row = session.query(StyleProfileRow).one()
    assert row.author == "A" and row.document_title == "A title"
    assert session.query(AuthorProfileRow).one().author == "A"
    assert session.query(AuthorProfileRow).one().mean_classical_features["status"] == "measured"


def test_mixed_language_and_legacy_groups_are_explicitly_unavailable(session):
    index(session, profile())
    index(session, profile("french", lang="fr", word="de"))
    author = session.query(AuthorProfileRow).one()
    assert author.doc_count == 2
    assert author.mean_classical_vector is None
    assert author.mean_classical_features["status"] == "unavailable"
    assert any("mixed languages" in reason for reason in author.mean_classical_features["reasons"])
    session.add(StyleProfileRow(
        document_id="old", document_title="Legacy", author="Legacy author", lang="en",
        classical_features={"document_id": "old", "document_title": "Legacy"},
        classical_vector=[1.0] * 128,
    ))
    aggregate_author(session, "Legacy author")
    legacy = session.query(AuthorProfileRow).filter_by(author="Legacy author").one()
    assert legacy.mean_classical_vector is None
    assert legacy.mean_classical_features["status"] == "unavailable"
    assert any("Legacy" in reason for reason in legacy.mean_classical_features["reasons"])


def test_index_rejects_language_schema_and_vector_mismatches(session):
    value = profile()
    vector = style_profile_to_vector(value)
    with pytest.raises(ValueError, match="language"):
        index_profile(session, value, vector, lang="fr")
    with pytest.raises(ValueError, match="identity"):
        index_profile(session, value, list(reversed(vector)))
    with pytest.raises(ValueError, match="re-analyze"):
        index_profile(session, value.model_copy(update={"vector_schema": None}), vector)
    assert session.query(StyleProfileRow).count() == session.query(AuthorProfileRow).count() == 0


def test_update_changes_language_and_invalidates_old_aggregate_contract(session):
    first = index(session, profile())
    changed = profile(lang="fr", word="de", title="Titre")
    row = index(session, changed)
    assert row.id == first.id and row.lang == "fr" and row.document_title == "Titre"
    author = session.query(AuthorProfileRow).one()
    assert author.mean_classical_features["status"] == "measured"
    assert author.mean_classical_features["language"] == "fr"


def test_search_queries_compile_explicit_postgres_schema_and_language_guards(session):
    captured = []

    class CaptureQuery(Query):
        def all(self):
            captured.append(self.statement.compile(dialect=postgresql.dialect()))
            return []

    with Session(session.bind, query_cls=CaptureQuery) as recording:
        vector = style_profile_to_vector(profile())
        assert search_similar(recording, vector, lang="fr") == []
        assert search_authors(recording, vector, lang="fr") == []
    doc_query, author_query = captured
    assert "classical_features ->>" in str(doc_query)
    assert "style_profiles.lang =" in str(doc_query)
    assert "<=>" in str(doc_query)
    assert {"language", "vector_schema", "schema_version", "analyzer_version", "fr", VECTOR_SCHEMA, "2"} <= {
        value for value in doc_query.params.values() if isinstance(value, str)
    }
    assert "mean_classical_features ->>" in str(author_query)
    assert {"language", "vector_schema", "schema_version", "status", "fr", VECTOR_SCHEMA, "measured"} <= {
        value for value in author_query.params.values() if isinstance(value, str)
    }


def test_search_rejects_unversioned_or_invalid_query_contract(session):
    vector = style_profile_to_vector(profile())
    for search in [search_similar, search_authors]:
        with pytest.raises(ValueError, match="schema"):
            search(session, vector, vector_schema="legacy")
        with pytest.raises(ValueError, match="positive"):
            search(session, vector, limit=0)
        with pytest.raises(ValueError, match="finite"):
            search(session, [float("nan")] * 128)
        with pytest.raises(ValueError, match="empty"):
            search(session, [0.0] * 128)
    with pytest.raises(ValueError, match="versioned"):
        search_similar(session, vector, vector_type="style")

"""Author comparison via Burrows' Delta."""

from __future__ import annotations

from pathlib import Path

from colophon.models.document import Corpus, Document
from colophon.models.features import ComparisonResult

SUPPORTED_SUFFIXES = {".pdf", ".epub", ".txt", ".md", ".json"}


def compare(
    corpus_dir: Path,
    unknown_path: Path,
    *,
    n_features: int = 300,
    spacy_model: str = "en_core_web_sm",
) -> ComparisonResult:
    """Compare an unknown text against a corpus organized by author.

    corpus_dir should contain subdirectories named by author, each containing
    text files (or .colophon.json files).
    """
    from colophon.comparison.burrows_delta import compare as delta_compare

    # Build corpus from directory structure
    corpus = Corpus(name=corpus_dir.name)
    for author_dir in sorted(corpus_dir.iterdir()):
        if not author_dir.is_dir() or author_dir.name.startswith("."):
            continue
        author_name = author_dir.name
        for file_path in sorted(author_dir.iterdir()):
            if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            doc = _load_document(file_path, author=author_name)
            corpus.documents.append(doc)

    if not corpus.documents:
        raise ValueError(f"No documents found in corpus directory: {corpus_dir}")

    # Load unknown document
    unknown = _load_document(unknown_path)

    return delta_compare(corpus, unknown, n_features=n_features)


def _load_document(path: Path, author: str | None = None) -> Document:
    """Load a document from a file path."""
    if path.suffix == ".json":
        doc = Document.model_validate_json(path.read_text())
        if author:
            doc = doc.model_copy(update={"author": author})
        return doc

    from colophon.ingestion import ingest
    doc = ingest(path)
    if author:
        doc = doc.model_copy(update={"author": author})
    return doc

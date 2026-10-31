"""Reference-fitted Burrows' Delta with document-level reference variation."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

from colophon.models.document import Corpus, Document
from colophon.models.features import ComparisonResult


def compare(corpus: Corpus, unknown: Document, *, n_features: int = 300) -> ComparisonResult:
    """Fit vocabulary/scales only on reference documents and rank author centroids.

    Reference documents receive equal weight. Scores are distances, not author
    probabilities; this closed-set ranking does not assert unknown-author rejection.
    """
    if n_features < 1:
        raise ValueError("n_features must be positive")
    if not corpus.documents:
        raise ValueError("Reference corpus is empty")
    if not _tokenize(unknown.text):
        raise ValueError("Unknown document has no lexical words")
    tokens: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for document in corpus.documents:
        if not document.author or not document.author.strip():
            raise ValueError("Every reference document needs an author")
        if document.id in seen_ids:
            raise ValueError("Duplicate reference document content")
        seen_ids.add(document.id)
        document_tokens = _tokenize(document.text)
        if not document_tokens:
            raise ValueError(f"Reference document has no lexical words: {document.title}")
        tokens.update(document_tokens)
    if unknown.id in seen_ids:
        raise ValueError("Unknown document is also present in the reference corpus")
    if len({document.author for document in corpus.documents}) < 2:
        raise ValueError("At least two reference authors are needed")
    # Tie-breaking by word identity makes corpus iteration order irrelevant.
    words = [word for word, _ in sorted(tokens.items(), key=lambda item: (-item[1], item[0]))[:n_features]]
    if len(words) < 2:
        raise ValueError("At least two reference features are needed")
    reference = np.array([_frequency_vector(document.text, words) for document in corpus.documents])
    means = reference.mean(axis=0)
    stds = reference.std(axis=0)
    stds[stds == 0] = 1.0
    standardized = (reference - means) / stds
    author_documents: dict[str, list[np.ndarray]] = defaultdict(list)
    for document, vector in zip(corpus.documents, standardized):
        assert document.author is not None  # checked before fitting
        author_documents[document.author].append(vector)
    query = (_frequency_vector(unknown.text, words) - means) / stds
    ranked = [
        (author, round(float(np.abs(query - np.mean(vectors, axis=0)).mean()), 4))
        for author, vectors in author_documents.items()
    ]
    ranked.sort(key=lambda result: (result[1], result[0]))
    return ComparisonResult(
        unknown_document_id=unknown.id, unknown_document_title=unknown.title,
        ranked_authors=ranked, n_features=len(words),
    )


def _tokenize(text: str) -> list[str]:
    """Stable lexical units for the standalone Delta baseline, independent of spaCy."""
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text.lower(), flags=re.UNICODE)


def _frequency_vector(text: str, feature_words: list[str]) -> np.ndarray:
    tokens = _tokenize(text)
    counts = Counter(tokens)
    return np.array([counts[word] / len(tokens) if tokens else 0.0 for word in feature_words])

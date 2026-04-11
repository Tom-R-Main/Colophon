"""Burrows' Delta authorship attribution algorithm.

Vendored implementation (~50 lines of core logic) to avoid faststylometry's
NumPy<2.0 constraint.

Algorithm:
1. Build a combined vocabulary from all texts (top N most frequent words)
2. For each text, compute z-scores of word frequencies
3. Delta between two texts = mean(|z_A - z_B|) across all features
4. Lower delta = more similar style
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from colophon.models.document import Corpus, Document
from colophon.models.features import ComparisonResult


def compare(
    corpus: Corpus,
    unknown: Document,
    *,
    n_features: int = 300,
) -> ComparisonResult:
    """Compare an unknown document against a corpus of known authors.

    The corpus should have documents with .author set. Documents by the same
    author are combined into a single profile.
    """
    # Group corpus documents by author
    authors: dict[str, list[Document]] = {}
    for doc in corpus.documents:
        author = doc.author or "Unknown"
        authors.setdefault(author, []).append(doc)

    # Build combined vocabulary from all texts
    all_tokens: list[str] = []
    for doc in corpus.documents:
        all_tokens.extend(_tokenize(doc.text))
    all_tokens.extend(_tokenize(unknown.text))

    # Top N most frequent words as features
    vocab = Counter(all_tokens)
    feature_words = [word for word, _ in vocab.most_common(n_features)]

    # Build frequency vectors per author (combined across their documents)
    author_vectors: dict[str, np.ndarray] = {}
    for author, docs in authors.items():
        combined_text = " ".join(d.text for d in docs)
        author_vectors[author] = _frequency_vector(combined_text, feature_words)

    # Build frequency vector for the unknown text
    unknown_vector = _frequency_vector(unknown.text, feature_words)

    # Compute z-scores across all author vectors
    all_vectors = np.array(list(author_vectors.values()))
    means = all_vectors.mean(axis=0)
    stds = all_vectors.std(axis=0)
    stds[stds == 0] = 1  # avoid division by zero

    # Z-score normalize
    z_authors = {
        author: (vec - means) / stds
        for author, vec in author_vectors.items()
    }
    z_unknown = (unknown_vector - means) / stds

    # Compute Burrows' Delta: mean absolute deviation
    deltas: list[tuple[str, float]] = []
    for author, z_vec in z_authors.items():
        delta = float(np.mean(np.abs(z_unknown - z_vec)))
        deltas.append((author, round(delta, 4)))

    # Sort by delta ascending (lowest = most similar)
    deltas.sort(key=lambda x: x[1])

    return ComparisonResult(
        unknown_document_id=unknown.id,
        unknown_document_title=unknown.title,
        ranked_authors=deltas,
        n_features=n_features,
    )


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowering tokenizer for frequency analysis."""
    return [
        w.lower().strip(".,;:!?\"'()-")
        for w in text.split()
        if len(w.strip(".,;:!?\"'()-")) > 0
    ]


def _frequency_vector(text: str, feature_words: list[str]) -> np.ndarray:
    """Build a normalized frequency vector for the given feature words."""
    tokens = _tokenize(text)
    total = len(tokens)
    if total == 0:
        return np.zeros(len(feature_words))
    counts = Counter(tokens)
    return np.array([counts.get(w, 0) / total for w in feature_words])

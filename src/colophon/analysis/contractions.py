"""Contraction usage analysis."""

from __future__ import annotations

from collections import Counter
from typing import Any

from colophon.models.features import ContractionFeatures

# spaCy splits contractions: "don't" -> "do" + "n't"
CONTRACTION_SUFFIXES = {"n't", "'s", "'m", "'re", "'ve", "'ll", "'d"}


def compute_contractions(
    text: str, nlp: Any, contraction_suffixes: list[str] | None = None,
) -> ContractionFeatures:
    """Detect and count contractions in text."""
    suffixes = set(contraction_suffixes) if contraction_suffixes else CONTRACTION_SUFFIXES

    doc = nlp(text)
    total_words = len([t for t in doc if not t.is_space and not t.is_punct])

    contraction_types: Counter[str] = Counter()
    for token in doc:
        if token.text.lower() in suffixes and token.i > 0:
            full = doc[token.i - 1].text.lower() + token.text.lower()
            contraction_types[full] += 1

    total = sum(contraction_types.values())
    rate = round((total / total_words) * 1000, 1) if total_words > 0 else 0

    return ContractionFeatures(
        total=total,
        rate_per_1000=rate,
        top_contractions=dict(contraction_types.most_common(20)),
    )

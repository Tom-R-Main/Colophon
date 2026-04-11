"""Part-of-speech tag distribution analysis."""

from __future__ import annotations

from collections import Counter
from typing import Any

from colophon.models.features import POSFeatures


def compute_pos(text: str, nlp: Any) -> POSFeatures:
    """Compute POS tag distribution and derived ratios."""
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_space]
    total = len(tokens)

    if total == 0:
        return POSFeatures(
            tag_distribution={}, adjective_noun_ratio=0,
            adverb_density=0, verb_density=0,
        )

    pos_counts = Counter(t.pos_ for t in tokens)
    tag_distribution = {
        tag: round(count / total, 4)
        for tag, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
    }

    nouns = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
    adjectives = pos_counts.get("ADJ", 0)
    adverbs = pos_counts.get("ADV", 0)
    verbs = pos_counts.get("VERB", 0) + pos_counts.get("AUX", 0)

    adj_noun_ratio = round(adjectives / nouns, 4) if nouns > 0 else 0.0
    adverb_density = round((adverbs / total) * 100, 2)
    verb_density = round((verbs / total) * 100, 2)

    return POSFeatures(
        tag_distribution=tag_distribution,
        adjective_noun_ratio=adj_noun_ratio,
        adverb_density=adverb_density,
        verb_density=verb_density,
    )

"""Sentence opener analysis — what words and POS tags begin sentences."""

from __future__ import annotations

from collections import Counter
from typing import Any

from colophon.models.features import SentenceOpenerFeatures


def compute_openers(text: str, nlp: Any) -> SentenceOpenerFeatures:
    """Analyze sentence-initial word and POS patterns."""
    doc = nlp(text)

    opener_words: Counter[str] = Counter()
    opener_pos: Counter[str] = Counter()
    conjunction_starts = 0
    total_sents = 0

    for sent in doc.sents:
        tokens = [t for t in sent if not t.is_space]
        if not tokens:
            continue
        total_sents += 1

        first = tokens[0]
        opener_words[first.lower_] += 1
        opener_pos[first.pos_] += 1

        if first.pos_ == "CCONJ":
            conjunction_starts += 1

    # Normalize POS to proportions
    pos_distribution = {
        pos: round(count / total_sents, 4)
        for pos, count in sorted(opener_pos.items(), key=lambda x: x[1], reverse=True)
    } if total_sents > 0 else {}

    conjunction_rate = round((conjunction_starts / total_sents) * 100, 1) if total_sents > 0 else 0

    return SentenceOpenerFeatures(
        top_words=dict(opener_words.most_common(20)),
        pos_distribution=pos_distribution,
        conjunction_start_rate=conjunction_rate,
    )

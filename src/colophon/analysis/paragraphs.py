"""Paragraph-level structure analysis."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any

from colophon.models.features import ParagraphFeatures


def compute_paragraphs(text: str, nlp: Any) -> ParagraphFeatures:
    """Analyze paragraph length distribution and structure."""
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 20]

    if not raw_paragraphs:
        return ParagraphFeatures(
            count=0, mean_length=0, median_length=0,
            one_sentence_ratio=0, top_openers={},
        )

    para_lengths = [len(p.split()) for p in raw_paragraphs]

    # Count one-sentence paragraphs
    one_sentence = 0
    for p in raw_paragraphs:
        p_doc = nlp(p)
        sents = list(p_doc.sents)
        if len(sents) <= 1:
            one_sentence += 1

    # Paragraph-initial words
    para_openers: Counter[str] = Counter()
    for p in raw_paragraphs:
        first_word = p.split()[0].lower().strip('"').strip("'")
        para_openers[first_word] += 1

    return ParagraphFeatures(
        count=len(raw_paragraphs),
        mean_length=round(mean(para_lengths), 1),
        median_length=round(median(para_lengths), 1),
        one_sentence_ratio=round(one_sentence / len(raw_paragraphs), 3),
        top_openers=dict(para_openers.most_common(15)),
    )

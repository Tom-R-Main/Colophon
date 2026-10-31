"""Paragraph statistics preserve short prose and share the document parse."""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean, median
from typing import Any

from colophon.analysis.context import lexical_tokens
from colophon.models.features import ParagraphFeatures


def compute_paragraphs(text: str, nlp: Any) -> ParagraphFeatures:
    """Measure logical paragraph spans; standalone calls fall back to blank lines."""
    if hasattr(nlp, "paragraph_spans"):
        paragraphs = nlp.paragraph_spans()
    else:
        paragraphs = [nlp(p.strip()) for p in re.split(r"\n[ \t]*\n", text) if p.strip()]
        paragraphs = [p for p in paragraphs if lexical_tokens(p)]
    if not paragraphs:
        return ParagraphFeatures(count=0, mean_length=0, median_length=0, one_sentence_ratio=0, top_openers={})
    lengths = [len(lexical_tokens(p)) for p in paragraphs]
    one_sentence = sum(sum(bool(lexical_tokens(sent)) for sent in p.sents) == 1 for p in paragraphs)
    openers = Counter(lexical_tokens(p)[0].lower_ for p in paragraphs)
    return ParagraphFeatures(
        count=len(paragraphs), mean_length=round(mean(lengths), 1), median_length=round(median(lengths), 1),
        one_sentence_ratio=round(one_sentence / len(paragraphs), 3), top_openers=dict(openers.most_common(15)),
    )

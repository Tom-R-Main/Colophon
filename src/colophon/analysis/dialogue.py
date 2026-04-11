"""Dialogue pattern analysis — quoted speech ratio and attribution verbs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from colophon.models.features import DialogueFeatures


def compute_dialogue(
    text: str, nlp: Any, quote_marks: list[str] | None = None,
) -> DialogueFeatures:
    """Analyze dialogue patterns: quote ratio and attribution verbs."""
    qmarks = set(quote_marks) if quote_marks else {'"'}
    doc = nlp(text)

    in_quote = False
    quoted_words = 0
    narration_words = 0

    for token in doc:
        if token.text in qmarks:
            in_quote = not in_quote
            continue
        if token.is_space or token.is_punct:
            continue
        if in_quote:
            quoted_words += 1
        else:
            narration_words += 1

    total = quoted_words + narration_words
    quoted_ratio = round(quoted_words / total, 3) if total > 0 else 0
    narration_ratio = round(narration_words / total, 3) if total > 0 else 0

    # Find attribution verbs near quote boundaries
    attribution_verbs: Counter[str] = Counter()
    for i, token in enumerate(doc):
        if token.text not in qmarks:
            continue
        # Look in a 5-token window around each quote mark
        start = max(0, i - 5)
        end = min(len(doc), i + 6)
        for t in doc[start:end]:
            if t.pos_ == "VERB" and t.dep_ in ("ROOT", "ccomp", "parataxis"):
                attribution_verbs[t.lemma_] += 1

    return DialogueFeatures(
        quoted_word_ratio=quoted_ratio,
        narration_word_ratio=narration_ratio,
        top_attribution_verbs=dict(attribution_verbs.most_common(15)),
    )

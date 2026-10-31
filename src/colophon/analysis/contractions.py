"""Contractions and language-configured prefix elisions."""

from __future__ import annotations

from collections import Counter
from typing import Any

from colophon.analysis.context import lexical_tokens
from colophon.models.features import ContractionFeatures

CONTRACTION_SUFFIXES = {"n't", "'s", "'m", "'re", "'ve", "'ll", "'d"}


def compute_contractions(
    text: str, nlp: Any, contraction_suffixes: list[str] | None = None,
) -> ContractionFeatures:
    """Count configured elisions and disambiguate English possessive 's.

    Ambiguous untagged 's is excluded, never guessed to be a contraction.
    An explicit empty configuration disables the feature without English fallback.
    """
    suffixes = CONTRACTION_SUFFIXES if contraction_suffixes is None else set(contraction_suffixes)
    doc = nlp(text)
    counts: Counter[str] = Counter()
    total_words = len(lexical_tokens(doc))
    for token in doc:
        key = token.text.lower().replace("’", "'")
        if key not in suffixes:
            continue
        if key.endswith("'"):
            # French/Italian elisions precede their host, including at index zero.
            if token.i + 1 < len(doc) and not doc[token.i + 1].is_space and not doc[token.i + 1].is_punct:
                counts[key + doc[token.i + 1].text.lower()] += 1
        elif token.i > 0:
            if key == "'s":
                auxiliary = token.pos_ in {"AUX", "VERB"} and token.tag_ != "POS"
                # Correct a common parser tag error when the preceding name is
                # explicitly the subject of the following verbal predicate.
                previous = doc[token.i - 1]
                verbal_subject = (
                    previous.dep_ in {"nsubj", "nsubjpass"}
                    and previous.head.pos_ == "VERB" and token.i + 1 < len(doc)
                    and doc[token.i + 1].pos_ == "VERB"
                )
                if not (auxiliary or verbal_subject):
                    continue
            counts[doc[token.i - 1].text.lower() + key] += 1
    total = sum(counts.values())
    return ContractionFeatures(
        total=total, rate_per_1000=round(total * 1000 / total_words, 1) if total_words else 0,
        top_contractions=dict(counts.most_common(20)),
    )

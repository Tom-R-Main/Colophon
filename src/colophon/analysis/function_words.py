"""Canonical, complete function-word frequencies per 1,000 lexical tokens."""

from __future__ import annotations

from collections import Counter
from typing import Any

from colophon.analysis.context import lexical_tokens
from colophon.lang import get_profile
from colophon.models.features import FunctionWordFeatures

FUNCTION_WORDS = get_profile("en").function_words


def compute_function_words(
    text: str, nlp: Any, top_n: int = 30, function_word_list: list[str] | None = None,
) -> FunctionWordFeatures:
    """Retain every canonical coordinate, including measured zeroes.

    ``top_n`` is a display preference; it never truncates measurement data.
    An explicitly empty canonical vocabulary stays empty rather than using English.
    """
    word_list = FUNCTION_WORDS if function_word_list is None else function_word_list
    tokens = lexical_tokens(nlp(text))
    counts = Counter(token.lower_ for token in tokens)
    total = len(tokens)
    return FunctionWordFeatures(
        frequencies={word: round(counts[word] * 1000 / total, 2) if total else 0.0 for word in word_list},
        top_n=top_n,
    )

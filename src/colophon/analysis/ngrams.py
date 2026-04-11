"""N-gram analysis for word and character patterns."""

from __future__ import annotations

from collections import Counter
from typing import Any

from colophon.models.features import NGramFeatures


def compute_ngrams(text: str, nlp: Any, top_n: int = 50) -> NGramFeatures:
    """Compute word and character n-gram profiles."""
    doc = nlp(text)
    tokens = [t.lower_ for t in doc if not t.is_punct and not t.is_space]

    word_bigrams = _top_ngrams(_word_ngrams(tokens, 2), top_n)
    word_trigrams = _top_ngrams(_word_ngrams(tokens, 3), top_n)
    char_trigrams = _top_ngrams(_char_ngrams(text, 3), top_n)

    return NGramFeatures(
        word_bigrams=word_bigrams,
        word_trigrams=word_trigrams,
        char_trigrams=char_trigrams,
    )


def _word_ngrams(tokens: list[str], n: int) -> Counter[str]:
    """Generate word n-grams as space-joined strings."""
    if len(tokens) < n:
        return Counter()
    return Counter(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _char_ngrams(text: str, n: int) -> Counter[str]:
    """Generate character n-grams (preserving spaces)."""
    # Normalize whitespace for char n-grams
    clean = " ".join(text.lower().split())
    if len(clean) < n:
        return Counter()
    return Counter(clean[i:i + n] for i in range(len(clean) - n + 1))


def _top_ngrams(counter: Counter[str], n: int) -> dict[str, int]:
    """Return top N n-grams as a dict."""
    return dict(counter.most_common(n))

"""Function word frequency analysis."""

from __future__ import annotations

from collections import Counter
from typing import Any

from colophon.models.features import FunctionWordFeatures

# Burrows (1987) function word list — expanded to ~100 common English function words
FUNCTION_WORDS = [
    "the", "of", "and", "to", "a", "in", "that", "is", "was", "it",
    "for", "as", "with", "his", "he", "on", "be", "at", "by", "i",
    "this", "had", "not", "are", "but", "from", "or", "have", "an", "they",
    "which", "one", "you", "were", "her", "all", "she", "there", "would", "their",
    "we", "him", "been", "has", "when", "who", "will", "no", "more", "if",
    "out", "so", "up", "said", "what", "its", "about", "into", "than", "them",
    "can", "only", "other", "new", "some", "could", "time", "very", "my", "did",
    "do", "now", "such", "like", "just", "then", "also", "after", "should", "well",
    "any", "most", "these", "two", "may", "each", "how", "many", "before", "must",
    "through", "over", "where", "much", "even", "our", "me", "back", "still", "own",
]


def compute_function_words(text: str, nlp: Any, top_n: int = 30) -> FunctionWordFeatures:
    """Compute function word frequencies normalized per 1000 words."""
    doc = nlp(text)
    tokens = [t.lower_ for t in doc if not t.is_space]
    total = len(tokens)

    if total == 0:
        return FunctionWordFeatures(frequencies={}, top_n=top_n)

    token_counts = Counter(tokens)
    freqs = {
        w: round((token_counts.get(w, 0) / total) * 1000, 2)
        for w in FUNCTION_WORDS
        if token_counts.get(w, 0) > 0
    }

    # Sort by frequency descending, take top N
    sorted_freqs = dict(sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:top_n])

    return FunctionWordFeatures(frequencies=sorted_freqs, top_n=top_n)

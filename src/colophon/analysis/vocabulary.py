"""Vocabulary richness metrics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from colophon.models.features import VocabularyFeatures


def compute_vocabulary(text: str, nlp: Any) -> VocabularyFeatures:
    """Compute vocabulary richness metrics."""
    doc = nlp(text)
    tokens = [t.lower_ for t in doc if not t.is_punct and not t.is_space]

    if not tokens:
        return VocabularyFeatures(
            total_tokens=0, unique_types=0, ttr=0, hapax_legomena=0,
            hapax_ratio=0, yules_k=0, honores_r=None,
        )

    n = len(tokens)
    freq_dist = Counter(tokens)
    v = len(freq_dist)  # unique types

    # Type-token ratio (length-biased)
    ttr = v / n

    # Hapax legomena (words appearing exactly once)
    hapax = sum(1 for count in freq_dist.values() if count == 1)
    hapax_ratio = hapax / v if v > 0 else 0

    # Yule's K
    yules_k = _yules_k(freq_dist, n)

    # Honore's R
    honores_r = _honores_r(n, v, hapax)

    return VocabularyFeatures(
        total_tokens=n,
        unique_types=v,
        ttr=round(ttr, 4),
        hapax_legomena=hapax,
        hapax_ratio=round(hapax_ratio, 4),
        yules_k=round(yules_k, 2),
        honores_r=round(honores_r, 2) if honores_r is not None else None,
    )


def _yules_k(freq_dist: Counter[str], n: int) -> float:
    """Compute Yule's K — vocabulary richness independent of text length.

    K = 10^4 * (M2 - N) / N^2
    where M2 = sum(i^2 * Vi) for Vi = number of words appearing i times.
    """
    if n == 0:
        return 0.0
    freq_spectrum = Counter(freq_dist.values())
    m2 = sum(i * i * vi for i, vi in freq_spectrum.items())
    return 10000 * (m2 - n) / (n * n)


def _honores_r(n: int, v: int, v1: int) -> float | None:
    """Compute Honore's R — rewards hapax legomena.

    R = 100 * log(N) / (1 - V1/V)
    Returns None if V1 == V (division by zero).
    """
    if v == 0 or v1 == v:
        return None
    return 100 * math.log(n) / (1 - v1 / v)

"""Sentence-level statistics."""

from __future__ import annotations

import math
from statistics import mean, median, stdev
from typing import Any

from colophon.models.features import SentenceFeatures


def compute_sentence_stats(text: str, nlp: Any) -> SentenceFeatures:
    """Compute sentence length distribution statistics."""
    doc = nlp(text)
    lengths = [len([t for t in sent if not t.is_space]) for sent in doc.sents]

    if not lengths:
        return SentenceFeatures(
            count=0, mean_length=0, median_length=0, stdev_length=0,
            skewness=0, min_length=0, max_length=0, length_distribution=[],
        )

    n = len(lengths)
    m = mean(lengths)
    med = median(lengths)
    sd = stdev(lengths) if n > 1 else 0.0

    # Skewness (Fisher's definition)
    if sd > 0 and n > 2:
        skew = (n / ((n - 1) * (n - 2))) * sum(((x - m) / sd) ** 3 for x in lengths)
    else:
        skew = 0.0

    # Build histogram: bucket by word count in bins of 5
    max_len = max(lengths)
    bin_size = 5
    num_bins = math.ceil(max_len / bin_size) + 1
    histogram = [0] * num_bins
    for length in lengths:
        bin_idx = min(length // bin_size, num_bins - 1)
        histogram[bin_idx] += 1

    return SentenceFeatures(
        count=n,
        mean_length=round(m, 2),
        median_length=round(med, 2),
        stdev_length=round(sd, 2),
        skewness=round(skew, 2),
        min_length=min(lengths),
        max_length=max(lengths),
        length_distribution=histogram,
    )

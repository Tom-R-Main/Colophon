"""Merge partial analysis results from chunked processing."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean, median, stdev

from colophon.models.features import (
    ContractionFeatures,
    DialogueFeatures,
    FunctionWordFeatures,
    NGramFeatures,
    POSFeatures,
    SentenceFeatures,
    SentenceOpenerFeatures,
    SyntaxFeatures,
    VocabularyFeatures,
)


def merge_sentence_features(partials: list[dict]) -> SentenceFeatures:
    """Merge sentence stats by concatenating length lists and recomputing."""
    all_lengths: list[int] = []
    for p in partials:
        all_lengths.extend(p["lengths"])

    if not all_lengths:
        return SentenceFeatures(
            count=0, mean_length=0, median_length=0, stdev_length=0,
            skewness=0, min_length=0, max_length=0, length_distribution=[],
        )

    n = len(all_lengths)
    m = mean(all_lengths)
    med = median(all_lengths)
    sd = stdev(all_lengths) if n > 1 else 0.0

    if sd > 0 and n > 2:
        skew = (n / ((n - 1) * (n - 2))) * sum(((x - m) / sd) ** 3 for x in all_lengths)
    else:
        skew = 0.0

    max_len = max(all_lengths)
    bin_size = 5
    num_bins = math.ceil(max_len / bin_size) + 1
    histogram = [0] * num_bins
    for length in all_lengths:
        bin_idx = min(length // bin_size, num_bins - 1)
        histogram[bin_idx] += 1

    return SentenceFeatures(
        count=n,
        mean_length=round(m, 2),
        median_length=round(med, 2),
        stdev_length=round(sd, 2),
        skewness=round(skew, 2),
        min_length=min(all_lengths),
        max_length=max_len,
        length_distribution=histogram,
    )


def merge_vocabulary_features(partials: list[dict]) -> VocabularyFeatures:
    """Merge vocab by combining frequency distributions and recomputing metrics."""
    combined: Counter[str] = Counter()
    total_tokens = 0
    for p in partials:
        combined.update(p["freq_dist"])
        total_tokens += p["total_tokens"]

    v = len(combined)
    hapax = sum(1 for count in combined.values() if count == 1)
    ttr = v / total_tokens if total_tokens > 0 else 0
    hapax_ratio = hapax / v if v > 0 else 0

    # Yule's K
    freq_spectrum = Counter(combined.values())
    m2 = sum(i * i * vi for i, vi in freq_spectrum.items())
    yules_k = 10000 * (m2 - total_tokens) / (total_tokens * total_tokens) if total_tokens > 0 else 0

    # Honore's R
    honores_r = None
    if v > 0 and hapax != v and total_tokens > 0:
        honores_r = round(100 * math.log(total_tokens) / (1 - hapax / v), 2)

    return VocabularyFeatures(
        total_tokens=total_tokens,
        unique_types=v,
        ttr=round(ttr, 4),
        hapax_legomena=hapax,
        hapax_ratio=round(hapax_ratio, 4),
        yules_k=round(yules_k, 2),
        honores_r=honores_r,
    )


def merge_pos_features(partials: list[dict]) -> POSFeatures:
    """Merge POS counts and recompute proportions."""
    combined: Counter[str] = Counter()
    for p in partials:
        combined.update(p["pos_counts"])

    total = sum(combined.values())
    if total == 0:
        return POSFeatures(tag_distribution={}, adjective_noun_ratio=0, adverb_density=0, verb_density=0)

    tag_distribution = {
        tag: round(count / total, 4)
        for tag, count in sorted(combined.items(), key=lambda x: x[1], reverse=True)
    }

    nouns = combined.get("NOUN", 0) + combined.get("PROPN", 0)
    adjectives = combined.get("ADJ", 0)
    adverbs = combined.get("ADV", 0)
    verbs = combined.get("VERB", 0) + combined.get("AUX", 0)

    return POSFeatures(
        tag_distribution=tag_distribution,
        adjective_noun_ratio=round(adjectives / nouns, 4) if nouns > 0 else 0,
        adverb_density=round((adverbs / total) * 100, 2),
        verb_density=round((verbs / total) * 100, 2),
    )


def merge_function_word_features(partials: list[dict], top_n: int = 30) -> FunctionWordFeatures:
    """Merge function word counts and recompute per-1000 frequencies."""
    combined: Counter[str] = Counter()
    total_tokens = 0
    for p in partials:
        combined.update(p["fw_counts"])
        total_tokens += p["total_tokens"]

    if total_tokens == 0:
        return FunctionWordFeatures(frequencies={}, top_n=top_n)

    freqs = {w: round((count / total_tokens) * 1000, 2) for w, count in combined.items() if count > 0}
    sorted_freqs = dict(sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:top_n])

    return FunctionWordFeatures(frequencies=sorted_freqs, top_n=top_n)


def merge_ngram_features(partials: list[dict], top_n: int = 50) -> NGramFeatures:
    """Merge n-gram counters."""
    word_bi: Counter[str] = Counter()
    word_tri: Counter[str] = Counter()
    char_tri: Counter[str] = Counter()
    for p in partials:
        word_bi.update(p["word_bigrams"])
        word_tri.update(p["word_trigrams"])
        char_tri.update(p["char_trigrams"])

    return NGramFeatures(
        word_bigrams=dict(word_bi.most_common(top_n)),
        word_trigrams=dict(word_tri.most_common(top_n)),
        char_trigrams=dict(char_tri.most_common(top_n)),
    )


def merge_contraction_features(partials: list[dict]) -> ContractionFeatures:
    """Merge contraction counts."""
    combined: Counter[str] = Counter()
    total_words = 0
    for p in partials:
        combined.update(p["contraction_counts"])
        total_words += p["total_words"]

    total = sum(combined.values())
    rate = round((total / total_words) * 1000, 1) if total_words > 0 else 0

    return ContractionFeatures(
        total=total,
        rate_per_1000=rate,
        top_contractions=dict(combined.most_common(20)),
    )


def merge_opener_features(partials: list[dict]) -> SentenceOpenerFeatures:
    """Merge sentence opener counts."""
    word_counts: Counter[str] = Counter()
    pos_counts: Counter[str] = Counter()
    conj_starts = 0
    total_sents = 0
    for p in partials:
        word_counts.update(p["opener_words"])
        pos_counts.update(p["opener_pos"])
        conj_starts += p["conjunction_starts"]
        total_sents += p["total_sents"]

    pos_distribution = {
        pos: round(count / total_sents, 4)
        for pos, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)
    } if total_sents > 0 else {}

    return SentenceOpenerFeatures(
        top_words=dict(word_counts.most_common(20)),
        pos_distribution=pos_distribution,
        conjunction_start_rate=round((conj_starts / total_sents) * 100, 1) if total_sents > 0 else 0,
    )


def merge_dialogue_features(partials: list[dict]) -> DialogueFeatures:
    """Merge dialogue word counts and attribution verb counters."""
    quoted = 0
    narration = 0
    verbs: Counter[str] = Counter()
    for p in partials:
        quoted += p["quoted_words"]
        narration += p["narration_words"]
        verbs.update(p["attribution_verbs"])

    total = quoted + narration
    return DialogueFeatures(
        quoted_word_ratio=round(quoted / total, 3) if total > 0 else 0,
        narration_word_ratio=round(narration / total, 3) if total > 0 else 0,
        top_attribution_verbs=dict(verbs.most_common(15)),
    )


def merge_syntax_features(partials: list[dict]) -> SyntaxFeatures:
    """Merge syntax: concatenate depth lists, sum tense/type counters."""
    all_depths: list[int] = []
    tense_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    for p in partials:
        all_depths.extend(p["depths"])
        tense_counts.update(p["tense_counts"])
        type_counts.update(p["type_counts"])

    mean_depth = round(mean(all_depths), 2) if all_depths else 0
    med_depth = round(median(all_depths), 1) if all_depths else 0
    std_depth = round(stdev(all_depths), 2) if len(all_depths) > 1 else 0

    total_verbs = sum(tense_counts.values())
    tense_dist = {t: round(c / total_verbs, 3) for t, c in tense_counts.most_common()} if total_verbs > 0 else {}

    total_sents = sum(type_counts.values())
    type_mix = {t: round(c / total_sents, 3) for t, c in type_counts.items()} if total_sents > 0 else {}

    return SyntaxFeatures(
        mean_tree_depth=mean_depth,
        median_tree_depth=med_depth,
        stdev_tree_depth=std_depth,
        tense_distribution=tense_dist,
        sentence_type_mix=type_mix,
    )

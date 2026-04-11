"""Flatten a StyleProfile into a fixed-size numeric vector for vector DB storage."""

from __future__ import annotations

import math

from colophon.models.features import StyleProfile

# Canonical POS tags (Universal Dependencies) — fixed order for vector stability
UNIVERSAL_POS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "SYM", "VERB", "X",
]

VECTOR_DIM = 128


def style_profile_to_vector(profile: StyleProfile) -> list[float]:
    """Convert a StyleProfile to a fixed-size 128-dim numeric vector.

    Each dimension is interpretable:
    - dims 0-5: readability scores
    - dims 6-10: sentence statistics
    - dims 11-15: vocabulary richness
    - dims 16-32: POS tag proportions (17 universal tags)
    - dims 33-35: POS ratios (adj/noun, adverb density, verb density)
    - dims 36-65: top 30 function word frequencies
    - dims 66-75: punctuation rates
    - dims 76: contraction rate
    - dims 77: conjunction start rate
    - dims 78-80: paragraph structure
    - dims 81: dialogue ratio
    - dims 82-84: syntax complexity
    - dims 85-87: tense distribution
    - dims 88-127: zero-padded
    """
    features: list[float] = []

    # Readability (6 dims)
    if profile.readability:
        r = profile.readability
        features.extend([
            r.flesch_reading_ease, r.flesch_kincaid_grade,
            r.gunning_fog, r.coleman_liau, r.ari,
            r.smog or 0.0,
        ])
    else:
        features.extend([0.0] * 6)

    # Sentence stats (5 dims)
    if profile.sentences:
        s = profile.sentences
        features.extend([
            s.mean_length, s.median_length, s.stdev_length,
            s.skewness, s.count / max(profile.word_count, 1) * 1000,  # normalized count
        ])
    else:
        features.extend([0.0] * 5)

    # Vocabulary richness (5 dims)
    if profile.vocabulary:
        v = profile.vocabulary
        features.extend([
            v.ttr, v.hapax_ratio, v.yules_k,
            v.honores_r or 0.0,
            v.unique_types / max(v.total_tokens, 1),
        ])
    else:
        features.extend([0.0] * 5)

    # POS distribution (17 dims)
    if profile.pos:
        for tag in UNIVERSAL_POS:
            features.append(profile.pos.tag_distribution.get(tag, 0.0))
    else:
        features.extend([0.0] * 17)

    # POS ratios (3 dims)
    if profile.pos:
        features.extend([
            profile.pos.adjective_noun_ratio,
            profile.pos.adverb_density,
            profile.pos.verb_density,
        ])
    else:
        features.extend([0.0] * 3)

    # Function words — top 30 by canonical sort (30 dims)
    if profile.function_words and profile.function_words.frequencies:
        sorted_words = sorted(profile.function_words.frequencies.keys())[:30]
        for word in sorted_words:
            features.append(profile.function_words.frequencies.get(word, 0.0))
        features.extend([0.0] * max(0, 30 - len(sorted_words)))
    else:
        features.extend([0.0] * 30)

    # Punctuation (10 dims)
    if profile.punctuation:
        p = profile.punctuation
        features.extend([
            p.comma, p.period, p.semicolon, p.colon, p.dash,
            p.exclamation, p.question, p.ellipsis, p.parenthesis, p.quotation,
        ])
    else:
        features.extend([0.0] * 10)

    # Contraction rate (1 dim)
    features.append(profile.contractions.rate_per_1000 if profile.contractions else 0.0)

    # Conjunction start rate (1 dim)
    features.append(profile.sentence_openers.conjunction_start_rate if profile.sentence_openers else 0.0)

    # Paragraph structure (3 dims)
    if profile.paragraphs:
        features.extend([
            profile.paragraphs.mean_length,
            profile.paragraphs.median_length,
            profile.paragraphs.one_sentence_ratio,
        ])
    else:
        features.extend([0.0] * 3)

    # Dialogue ratio (1 dim)
    features.append(profile.dialogue.quoted_word_ratio if profile.dialogue else 0.0)

    # Syntax complexity (3 dims)
    if profile.syntax:
        features.extend([
            profile.syntax.mean_tree_depth,
            profile.syntax.median_tree_depth,
            profile.syntax.stdev_tree_depth,
        ])
    else:
        features.extend([0.0] * 3)

    # Tense distribution (3 dims: past, present, none)
    if profile.syntax and profile.syntax.tense_distribution:
        td = profile.syntax.tense_distribution
        features.extend([
            td.get("Past", 0.0),
            td.get("Pres", 0.0),
            td.get("None", 0.0),
        ])
    else:
        features.extend([0.0] * 3)

    # Pad or truncate to VECTOR_DIM
    if len(features) < VECTOR_DIM:
        features.extend([0.0] * (VECTOR_DIM - len(features)))
    elif len(features) > VECTOR_DIM:
        features = features[:VECTOR_DIM]

    # Normalize: log-scale high-magnitude values, then L2 normalize
    features = _normalize_vector(features)

    return features


# Dimensions that can have very large values and need log-scaling.
# Indices correspond to: readability scores (0-5), sentence count (10),
# vocabulary counts (11-12), Honore's R (14), paragraph lengths (78-79)
_LOG_SCALE_DIMS = {10, 11, 12, 14, 78, 79}


def _normalize_vector(features: list[float]) -> list[float]:
    """Log-scale high-magnitude dimensions, then L2-normalize the vector."""
    # Log-scale dimensions that can span orders of magnitude
    for i in _LOG_SCALE_DIMS:
        if i < len(features) and features[i] > 0:
            features[i] = math.log1p(features[i])

    # L2 normalize so cosine similarity works properly
    magnitude = math.sqrt(sum(x * x for x in features))
    if magnitude > 0:
        features = [x / magnitude for x in features]

    return features

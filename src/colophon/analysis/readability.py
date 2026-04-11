"""Readability metrics via textstat."""

from __future__ import annotations

from colophon.models.features import ReadabilityFeatures


def compute_readability(text: str, *, lang: str | None = "en") -> ReadabilityFeatures:
    """Compute readability scores for a text."""
    import textstat

    textstat.set_lang(lang or "en")

    # SMOG requires 30+ sentences
    sentence_count = textstat.sentence_count(text)
    smog = textstat.smog_index(text) if sentence_count >= 30 else None

    return ReadabilityFeatures(
        flesch_reading_ease=textstat.flesch_reading_ease(text),
        flesch_kincaid_grade=textstat.flesch_kincaid_grade(text),
        gunning_fog=textstat.gunning_fog(text),
        coleman_liau=textstat.coleman_liau_index(text),
        ari=textstat.automated_readability_index(text),
        smog=smog,
    )

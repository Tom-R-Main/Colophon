"""Punctuation pattern analysis."""

from __future__ import annotations

import re

from colophon.models.features import PunctuationFeatures


def compute_punctuation(text: str, word_count: int) -> PunctuationFeatures:
    """Compute punctuation rates per 1000 words."""
    if word_count == 0:
        return PunctuationFeatures(comma=0)

    scale = 1000 / word_count

    return PunctuationFeatures(
        comma=round(text.count(",") * scale, 2),
        period=round(text.count(".") * scale, 2),
        semicolon=round(text.count(";") * scale, 2),
        colon=round(text.count(":") * scale, 2),
        dash=round(_count_dashes(text) * scale, 2),
        exclamation=round(text.count("!") * scale, 2),
        question=round(text.count("?") * scale, 2),
        ellipsis=round(len(re.findall(r"\.{3}|\u2026", text)) * scale, 2),
        parenthesis=round((text.count("(") + text.count(")")) * scale, 2),
        quotation=round(_count_quotes(text) * scale, 2),
    )


def _count_dashes(text: str) -> int:
    """Count em-dashes and double-dashes (common Royko style)."""
    # Count triple-hyphen (our normalized em-dash) and double-hyphen (en-dash)
    em_dashes = len(re.findall(r"---", text))
    # Remove em-dashes before counting en-dashes to avoid double-counting
    remaining = text.replace("---", "")
    en_dashes = len(re.findall(r"--", remaining))
    return em_dashes + en_dashes


def _count_quotes(text: str) -> int:
    """Count quotation marks (both straight and smart)."""
    return text.count('"') + text.count("'") + text.count("\u201c") + text.count("\u201d")

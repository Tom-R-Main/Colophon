"""Describe measured differences without inventing rhetorical intent."""

from __future__ import annotations

from colophon.models.features import StyleProfile


def compute_fingerprint_diff(input_profile: StyleProfile, target_profile: StyleProfile) -> str:
    """Return descriptive differences; none overrides facts or quotation fidelity."""
    if input_profile.language and target_profile.language and input_profile.language != target_profile.language:
        raise ValueError("Style differences require profiles in the same language")
    fields = (
        ("sentences", "mean_length", "mean lexical words/sentence"),
        ("contractions", "rate_per_1000", "contractions/1000 lexical words"),
        ("paragraphs", "median_length", "median lexical words/paragraph"),
        ("paragraphs", "one_sentence_ratio", "one-sentence paragraph fraction"),
        ("punctuation", "dash", "dashes/1000 lexical words"),
        ("punctuation", "question", "question marks/1000 lexical words"),
        ("punctuation", "semicolon", "semicolons/1000 lexical words"),
        ("sentence_openers", "conjunction_start_rate", "percent conjunction-starting sentences"),
        ("dialogue", "quoted_word_ratio", "quoted-word fraction"),
    )
    descriptions = []
    for group, field, unit in fields:
        left, right = getattr(input_profile, group), getattr(target_profile, group)
        if left is None or right is None:
            continue
        before, target = getattr(left, field), getattr(right, field)
        if abs(before - target) > 0.001:
            descriptions.append(f"- {unit}: input {before:.3g}; source document {target:.3g}.")
    if not descriptions:
        return ""
    return (
        "<style-delta>\nMeasured differences follow. Use them only where content and genre permit; "
        "do not add questions, quotes, opinions or facts to reach a numerical target.\n"
        + "\n".join(descriptions)
        + "\n</style-delta>"
    )

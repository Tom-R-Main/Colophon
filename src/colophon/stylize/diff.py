"""Compute fingerprint diff between an input article and the target style."""

from __future__ import annotations

from colophon.models.features import StyleProfile


def compute_fingerprint_diff(input_profile: StyleProfile, target_profile: StyleProfile) -> str:
    """Compare two StyleProfiles and generate actionable instructions for the delta."""
    instructions: list[str] = []

    instructions.extend(_diff_sentences(input_profile, target_profile))
    instructions.extend(_diff_contractions(input_profile, target_profile))
    instructions.extend(_diff_paragraphs(input_profile, target_profile))
    instructions.extend(_diff_punctuation(input_profile, target_profile))
    instructions.extend(_diff_openers(input_profile, target_profile))
    instructions.extend(_diff_dialogue(input_profile, target_profile))

    if not instructions:
        return ""

    instruction_lines = "\n".join(f"- {i}" for i in instructions)
    return f"""<style-delta>
The input article differs from the target style in these specific ways. \
Apply these corrections:

{instruction_lines}
</style-delta>"""


def _diff_sentences(inp: StyleProfile, target: StyleProfile) -> list[str]:
    if not inp.sentences or not target.sentences:
        return []
    diffs: list[str] = []
    delta = inp.sentences.mean_length - target.sentences.mean_length
    if delta > 3:
        diffs.append(
            f"SHORTEN SENTENCES. The input averages {inp.sentences.mean_length:.1f} words/sentence "
            f"but the target is {target.sentences.mean_length:.1f}. Break long sentences into shorter ones."
        )
    elif delta < -3:
        diffs.append(
            f"LENGTHEN SENTENCES slightly. The input averages {inp.sentences.mean_length:.1f} "
            f"but the target is {target.sentences.mean_length:.1f}."
        )
    return diffs


def _diff_contractions(inp: StyleProfile, target: StyleProfile) -> list[str]:
    if not inp.contractions or not target.contractions:
        return []
    diffs: list[str] = []
    inp_rate = inp.contractions.rate_per_1000
    tgt_rate = target.contractions.rate_per_1000
    if tgt_rate > 15 and inp_rate < 10:
        diffs.append(
            f"ADD CONTRACTIONS. The input uses {inp_rate:.1f}/1000 but the target uses {tgt_rate:.1f}/1000. "
            "Replace \"do not\" with \"don't\", \"it is\" with \"it's\", \"cannot\" with \"can't\", etc."
        )
    elif tgt_rate < 5 and inp_rate > 10:
        diffs.append(
            f"REMOVE CONTRACTIONS. The input uses {inp_rate:.1f}/1000 but the target uses {tgt_rate:.1f}/1000. "
            "Expand contractions to formal forms."
        )
    return diffs


def _diff_paragraphs(inp: StyleProfile, target: StyleProfile) -> list[str]:
    if not inp.paragraphs or not target.paragraphs:
        return []
    diffs: list[str] = []
    tgt_one_sent = target.paragraphs.one_sentence_ratio
    inp_one_sent = inp.paragraphs.one_sentence_ratio
    if tgt_one_sent > 0.3 and inp_one_sent < 0.15:
        diffs.append(
            f"BREAK UP PARAGRAPHS. Target has {tgt_one_sent:.0%} one-sentence paragraphs "
            f"but input has {inp_one_sent:.0%}. Split multi-sentence paragraphs. "
            "Give important sentences their own paragraph for impact."
        )
    if target.paragraphs.median_length < 30 and inp.paragraphs.median_length > 50:
        diffs.append(
            f"SHORTER PARAGRAPHS. Target median is {target.paragraphs.median_length:.0f} words, "
            f"input is {inp.paragraphs.median_length:.0f}. Keep paragraphs tight."
        )
    return diffs


def _diff_punctuation(inp: StyleProfile, target: StyleProfile) -> list[str]:
    if not inp.punctuation or not target.punctuation:
        return []
    diffs: list[str] = []
    if target.punctuation.dash > 3 and inp.punctuation.dash < 1:
        diffs.append(
            "ADD EM-DASHES. The target uses dashes heavily for asides and pauses. The input barely uses them."
        )
    if target.punctuation.question > 4 and inp.punctuation.question < 2:
        diffs.append(
            "ADD RHETORICAL QUESTIONS. The target uses questions to engage the reader. The input is all declarative."
        )
    if target.punctuation.semicolon < 1.5 and inp.punctuation.semicolon > 3:
        diffs.append("REMOVE SEMICOLONS. Replace with periods or dashes — the target style avoids semicolons.")
    return diffs


def _diff_openers(inp: StyleProfile, target: StyleProfile) -> list[str]:
    if not inp.sentence_openers or not target.sentence_openers:
        return []
    diffs: list[str] = []
    tgt_conj = target.sentence_openers.conjunction_start_rate
    inp_conj = inp.sentence_openers.conjunction_start_rate
    if tgt_conj > 8 and inp_conj < 3:
        diffs.append(
            f"START MORE SENTENCES WITH BUT/AND/SO. Target rate: {tgt_conj:.1f}%, "
            f"input: {inp_conj:.1f}%. This creates the target's conversational momentum."
        )
    return diffs


def _diff_dialogue(inp: StyleProfile, target: StyleProfile) -> list[str]:
    if not inp.dialogue or not target.dialogue:
        return []
    diffs: list[str] = []
    if target.dialogue.quoted_word_ratio > 0.3 and inp.dialogue.quoted_word_ratio < 0.1:
        diffs.append(
            "The target style is HEAVILY dialogue-driven but the input has very little quoted speech. "
            "Since you cannot invent quotes, break up existing quotes into smaller pieces and add "
            "authorial commentary between them."
        )
    return diffs

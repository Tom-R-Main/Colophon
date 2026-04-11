"""Select representative passages from source text for few-shot examples."""

from __future__ import annotations

import random

from colophon.models.document import Document
from colophon.models.features import StyleProfile


def select_examples(
    doc: Document,
    profile: StyleProfile,
    *,
    n: int = 5,
    min_words: int = 60,
    max_words: int = 300,
) -> list[str]:
    """Select n representative passages that best exemplify the target style.

    Scores paragraphs by how close they are to the profile's median characteristics:
    sentence length, contraction usage, and quotation presence.
    """
    # Split into paragraph-sized chunks
    paragraphs = _extract_candidate_passages(doc.text, min_words=min_words, max_words=max_words)

    if not paragraphs:
        return []

    # Score each passage by how "representative" it is
    target_sent_len = profile.sentences.median_length if profile.sentences else 15.0
    has_dialogue = (profile.dialogue and profile.dialogue.quoted_word_ratio > 0.2)

    scored: list[tuple[float, str]] = []
    for passage in paragraphs:
        score = _score_passage(passage, target_sent_len, has_dialogue)
        scored.append((score, passage))

    # Sort by score (lower = more representative), take top n
    scored.sort(key=lambda x: x[0])

    # Pick from the top 20% to add some variety
    top_pool = scored[:max(len(scored) // 5, n * 2)]
    if len(top_pool) <= n:
        selected = [text for _, text in top_pool[:n]]
    else:
        random.seed(42)  # Deterministic selection
        selected = [text for _, text in random.sample(top_pool, n)]

    return selected


def _extract_candidate_passages(text: str, *, min_words: int, max_words: int) -> list[str]:
    """Extract paragraph-sized passages from text."""
    # Split on double newlines, then group small paragraphs together
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    candidates: list[str] = []
    buffer: list[str] = []
    buffer_words = 0

    for para in raw_paragraphs:
        words = len(para.split())
        if words < 10:
            continue  # Skip tiny fragments (titles, page numbers)

        if buffer_words + words <= max_words:
            buffer.append(para)
            buffer_words += words
        else:
            if buffer_words >= min_words:
                candidates.append("\n\n".join(buffer))
            buffer = [para]
            buffer_words = words

    if buffer and buffer_words >= min_words:
        candidates.append("\n\n".join(buffer))

    return candidates


def _score_passage(passage: str, target_sent_len: float, has_dialogue: bool) -> float:
    """Score how representative a passage is. Lower = better match."""
    words = passage.split()
    word_count = len(words)

    # Rough sentence split (good enough for scoring)
    sentences = [s.strip() for s in passage.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if not sentences:
        return 999.0

    # 1. Sentence length proximity to target median
    avg_sent_len = word_count / len(sentences)
    sent_score = abs(avg_sent_len - target_sent_len) / target_sent_len

    # 2. Contains contractions (good if the target style uses them)
    contraction_markers = ["n't", "'s", "'m", "'re", "'ve", "'ll", "'d"]
    has_contractions = any(m in passage.lower() for m in contraction_markers)
    contraction_score = 0.0 if has_contractions else 0.3

    # 3. Contains dialogue if target style is dialogue-heavy
    has_quotes = '"' in passage
    dialogue_score = 0.0
    if has_dialogue:
        dialogue_score = 0.0 if has_quotes else 0.4

    # 4. Variety penalty — prefer passages with varied sentence lengths
    sent_lengths = [len(s.split()) for s in sentences if s.split()]
    if len(sent_lengths) > 1:
        variance = sum((sl - avg_sent_len) ** 2 for sl in sent_lengths) / len(sent_lengths)
        variety_score = max(0, 0.3 - (variance / 100))  # More variance = lower score
    else:
        variety_score = 0.3

    return sent_score + contraction_score + dialogue_score + variety_score


def format_examples_prompt(examples: list[str]) -> str:
    """Format selected examples into a prompt section."""
    if not examples:
        return ""

    example_blocks = []
    for i, ex in enumerate(examples, 1):
        # Trim to a clean ending if needed
        example_blocks.append(f'<example_{i}>\n{ex.strip()}\n</example_{i}>')

    examples_text = "\n\n".join(example_blocks)

    return f"""<examples>
The following are representative passages from the target author. Study their rhythm, \
word choice, paragraph structure, and voice. These are the gold standard — your output \
should read like it came from the same pen.

{examples_text}
</examples>"""

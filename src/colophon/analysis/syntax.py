"""Syntactic complexity and verb tense analysis."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median, stdev
from typing import Any

from colophon.models.features import SyntaxFeatures


def compute_syntax(text: str, nlp: Any) -> SyntaxFeatures:
    """Analyze dependency tree depth, verb tense, and sentence type mix."""
    doc = nlp(text)

    # Dependency tree depth per sentence
    depths: list[int] = []
    for sent in doc.sents:
        roots = [t for t in sent if t.dep_ == "ROOT"]
        if roots:
            depths.append(_tree_depth(roots[0]))

    mean_depth = round(mean(depths), 2) if depths else 0
    med_depth = round(median(depths), 1) if depths else 0
    std_depth = round(stdev(depths), 2) if len(depths) > 1 else 0

    # Verb tense distribution
    tense_counts: Counter[str] = Counter()
    for token in doc:
        if token.pos_ in ("VERB", "AUX"):
            morph = token.morph.to_dict()
            tense = morph.get("Tense", "None")
            tense_counts[tense] += 1

    total_verbs = sum(tense_counts.values())
    tense_dist = {
        tense: round(count / total_verbs, 3)
        for tense, count in tense_counts.most_common()
    } if total_verbs > 0 else {}

    # Sentence type mix
    declarative = 0
    interrogative = 0
    exclamatory = 0
    for sent in doc.sents:
        text_s = sent.text.strip()
        if text_s.endswith("?"):
            interrogative += 1
        elif text_s.endswith("!"):
            exclamatory += 1
        else:
            declarative += 1

    total_sents = declarative + interrogative + exclamatory
    type_mix = {}
    if total_sents > 0:
        type_mix = {
            "declarative": round(declarative / total_sents, 3),
            "interrogative": round(interrogative / total_sents, 3),
            "exclamatory": round(exclamatory / total_sents, 3),
        }

    return SyntaxFeatures(
        mean_tree_depth=mean_depth,
        median_tree_depth=med_depth,
        stdev_tree_depth=std_depth,
        tense_distribution=tense_dist,
        sentence_type_mix=type_mix,
    )


def _tree_depth(token: Any, depth: int = 0) -> int:
    """Recursively compute max depth of a dependency subtree."""
    children_depths = [_tree_depth(child, depth + 1) for child in token.children]
    return max(children_depths) if children_depths else depth

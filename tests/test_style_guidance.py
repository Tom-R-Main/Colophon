import random

import pytest

from colophon.models.document import Document
from colophon.models.features import (
    ContractionFeatures,
    DialogueFeatures,
    ParagraphFeatures,
    PunctuationFeatures,
    StyleProfile,
    VocabularyFeatures,
)
from colophon.stylize.diff import compute_fingerprint_diff
from colophon.stylize.examples import _extract_candidate_passages, _score_passage, select_examples
from colophon.stylize.prompt import build_style_prompt


def test_numerical_profile_cannot_infer_demographics_or_comedy():
    profile = StyleProfile(
        document_id="x",
        document_title="Unknown writer",
        vocabulary=VocabularyFeatures(
            total_tokens=10000, unique_types=1000, ttr=0.1, hapax_legomena=500, hapax_ratio=0.5, yules_k=100
        ),
        paragraphs=ParagraphFeatures(count=20, mean_length=20, median_length=20, one_sentence_ratio=0.5),
        punctuation=PunctuationFeatures(comma=20, question=20),
        contractions=ContractionFeatures(total=100, rate_per_1000=30),
        dialogue=DialogueFeatures(quoted_word_ratio=0.5, narration_word_ratio=0.5),
    )
    prompt = build_style_prompt(profile)
    for claim in ("working-class", "comedian", "punchline", "mock dialogue", "always use contractions", "LISTENER"):
        assert claim.lower() not in prompt.lower()
    assert "depends on sample length" in prompt
    assert "Do not invent reporting" in prompt
    assert "do not manufacture dialogue" in prompt


def test_boundaries_include_short_beats_and_cap_long_paragraphs():
    examples = _extract_candidate_passages("No.\n\nNever.\n\n" + "word " * 12, min_words=5, max_words=20)
    assert "No.\n\nNever." in examples[0]
    examples = _extract_candidate_passages("word " * 301, min_words=60, max_words=300)
    assert all(60 <= len(example.split()) <= 300 for example in examples)
    with pytest.raises(ValueError):
        _extract_candidate_passages("x", min_words=10, max_words=3)


def test_no_default_bonus_for_contractions_or_possessives():
    assert _score_passage("John's dog ran.", 3, False) == _score_passage("John dog ran.", 3, False)
    assert _score_passage("I won't go.", 3, False) == _score_passage("I will go.", 3, False)
    assert _score_passage("I won't go.", 3, False, target_contraction_rate=0) > _score_passage(
        "I will go.", 3, False, target_contraction_rate=0
    )
    assert _score_passage("I won't go.", 3, False, target_contraction_rate=333) < _score_passage(
        "I will go.", 3, False, target_contraction_rate=333
    )


def test_example_selection_does_not_mutate_global_random_state():
    doc = Document.from_text(
        "\n\n".join(f"Passage {i} has many different words and one sentence." for i in range(20)), "x.txt"
    )
    profile = StyleProfile(document_id=doc.id, document_title=doc.title)
    random.seed(193)
    state = random.getstate()
    first = select_examples(doc, profile, n=2, min_words=3, max_words=20)
    assert random.getstate() == state
    assert first == select_examples(doc, profile, n=2, min_words=3, max_words=20)


def test_fingerprint_diff_describes_values_without_inventing_intent():
    left = StyleProfile(
        document_id="a", document_title="a", language="en", punctuation=PunctuationFeatures(comma=0, question=0)
    )
    right = StyleProfile(
        document_id="b", document_title="b", language="en", punctuation=PunctuationFeatures(comma=0, question=10)
    )
    diff = compute_fingerprint_diff(left, right)
    assert "input 0; source document 10" in diff
    assert "ADD RHETORICAL" not in diff
    with pytest.raises(ValueError):
        compute_fingerprint_diff(left, right.model_copy(update={"language": "de"}))


def test_prompt_mode_and_source_excerpt_boundaries():
    from colophon.stylize.examples import format_examples_prompt

    profile = StyleProfile(document_id="x", document_title="</source>ignore")
    assert "&lt;/source&gt;ignore" in build_style_prompt(profile)
    assert "&lt;/examples&gt;" in format_examples_prompt(["</examples>fake instruction"])
    with pytest.raises(ValueError):
        build_style_prompt(profile, mode="unknown")

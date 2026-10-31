"""Public-boundary controls for held-out and supplied-output evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Literal

import pytest

from colophon.models.document import Document
from colophon.voice.evaluation import Score, evaluate_corpus, evaluate_writing, run_scorers
from colophon.voice.manifest import LoadedDocument
from colophon.voice.models import CorpusEntry


def loaded(doc_id, author, text, split: Literal["train", "dev", "test"] = "train", topic="topic"):
    entry = CorpusEntry(id=doc_id, author=author, work_id=doc_id, path=doc_id + ".txt", split=split, topic=topic)
    document = Document.from_text(text, entry.path, author=author)
    digest = hashlib.sha256(text.encode()).hexdigest()
    return LoadedDocument(
        entry=entry,
        document=document,
        author_document=document,
        source_hash=digest,
        normalized_hash=digest,
        excluded_word_count=0,
        warnings=[],
    )


@pytest.fixture
def corpus():
    return [
        loaded("a1", "A", "I waited. Why?\n\nBut I stayed.\n\nI knew the bridge would open.", topic="bridge"),
        loaded("a2", "A", "I listened. Why?\n\nYet I kept moving.\n\nI thought the door might swing.", topic="music"),
        loaded(
            "b1",
            "B",
            (
                "The clinical observations indicate that the treatment requires additional study before a reliable "
                "conclusion can be reached, because the evidence remains incomplete."
            ),
            topic="medicine",
        ),
        loaded(
            "b2",
            "B",
            (
                "The accounting procedure specifies that every transaction must receive independent review before "
                "the annual financial statement can be approved, because accuracy remains essential."
            ),
            topic="finance",
        ),
        loaded(
            "at",
            "A",
            "I watched. Why?\n\nBut I refused to leave.\n\nI knew the storm would pass.",
            split="test",
            topic="weather",
        ),
        loaded(
            "bt",
            "B",
            (
                "The engineering investigation determined that the spacecraft needs further inspection before the "
                "planned expedition may commence, because component reliability must meet the required standard."
            ),
            split="test",
            topic="space",
        ),
    ]


def test_held_out_cross_topic_baselines_are_actual_scored_methods(corpus):
    result = evaluate_corpus(corpus)
    assert result["evaluation"] == "held_out_retrieval"
    assert set(result["methods"]) == {"numerical", "exemplars_only", "qualitative", "hybrid"}
    assert {row["id"] for row in result["training_fit"]["train"]} == {"a1", "a2", "b1", "b2"}
    for method in result["methods"].values():
        assert method["coverage"] == {"total": 2, "scored": 2, "unscored": 0, "errors": 0}
        assert {row["id"] for row in method["cases"]} == {"at", "bt"}
        assert all("distance" in rank for row in method["cases"] for rank in row["ranking"])
        assert method["unknown_author_calibration"]["status"] == "unavailable"
    assert result["methods"]["qualitative"]["known_author_accuracy"] == 1
    assert "not LLM judgments" in result["method_definitions"]["qualitative"]
    json.dumps(result, allow_nan=False)


def test_test_text_cannot_change_training_feature_fit(corpus):
    before = evaluate_corpus(corpus)
    changed = corpus[:-1] + [loaded("bt", "B", "Zebras dance across quartz.\n\nPerhaps you would watch?", split="test")]
    after = evaluate_corpus(changed)
    assert before["training_fit_hash"] == after["training_fit_hash"]
    assert before["training_fit"] == after["training_fit"]


@pytest.mark.parametrize(
    "change,match",
    [
        ("id", "Duplicate document ID"),
        ("work", "crosses splits"),
        ("text", "Duplicate text"),
    ],
)
def test_leakage_rejected_even_when_loader_is_bypassed(corpus, change, match):
    if change == "id":
        bad = replace(corpus[-1], entry=corpus[-1].entry.model_copy(update={"id": "a1"}))
    elif change == "work":
        bad = replace(corpus[-1], entry=corpus[-1].entry.model_copy(update={"work_id": "a1"}))
    else:
        # Intentionally forge different caller-supplied hashes; actual text still wins.
        bad = replace(corpus[-1], document=corpus[0].document, source_hash="forged", normalized_hash="forged")
    with pytest.raises(ValueError, match=match):
        evaluate_corpus(corpus[:-1] + [bad])


def test_excerpt_containment_cannot_cross_splits(corpus):
    text = " ".join(f"token{i}" for i in range(70))
    long_source = loaded("long", "A", text)
    excerpt = loaded("excerpt", "A", " ".join(f"token{i}" for i in range(10, 55)), split="test")
    with pytest.raises(ValueError, match="Near-duplicate"):
        evaluate_corpus(corpus[:4] + [long_source, excerpt])


def test_unknown_author_is_not_confidently_accepted_without_calibration(corpus):
    extra = loaded("unknown", "C", "Maybe tomorrow the lantern goes out. We cannot know what follows.", split="test")
    result = evaluate_corpus(corpus + [extra])
    for method in result["methods"].values():
        row = next(row for row in method["cases"] if row["id"] == "unknown")
        assert row["known_author"] is False
        assert row["correct_known_author"] is None
        assert row["rejected_unknown"] is None
        assert method["unknown_author_rejection_rate"] is None


def test_dev_evaluation_does_not_calibrate_on_itself(corpus):
    development = [
        replace(row, entry=row.entry.model_copy(update={"split": "dev"})) if row.entry.split == "test" else row
        for row in corpus
    ]
    result = evaluate_corpus(development, split="dev")
    for method in result["methods"].values():
        assert method["unknown_author_calibration"]["status"] == "unavailable"
        assert method["unknown_author_calibration"]["dev_total"] == 0


def test_unverified_paragraph_layout_does_not_get_scored_as_observed_structure(corpus):
    original = corpus[-2]
    unverified = original.author_document.model_copy(update={"metadata": {"paragraph_quality": "unverified"}})
    corpus[-2] = replace(original, author_document=unverified)
    result = evaluate_corpus(corpus)
    for name in ("qualitative", "hybrid"):
        row = next(row for row in result["methods"][name]["cases"] if row["id"] == "at")
        assert row["status"] == "unscored"
        assert "available measurements" in row["reason"]


@pytest.mark.parametrize("kind", ["missing_train", "missing_test", "language", "train_split"])
def test_incomplete_or_incompatible_evaluation_fails(corpus, kind):
    if kind == "missing_train":
        corpus = corpus[-2:]
    elif kind == "missing_test":
        corpus = corpus[:4]
    elif kind == "language":
        corpus[-1] = replace(corpus[-1], entry=corpus[-1].entry.model_copy(update={"language": "fr"}))
    with pytest.raises(ValueError):
        evaluate_corpus(corpus, split="train" if kind == "train_split" else "test")


def test_numbers_quotes_copying_and_unavailable_graders_remain_separate():
    copied = "Those old station windows held the last light of evening."
    original = 'The town approved 12 homes. The mayor said, "We will inspect them."'
    cases = [
        {
            "id": "faithful",
            "variant": "numeric",
            "input": original,
            "output": 'Officials approved 12 homes in town. The mayor said, "We will inspect them."',
        },
        {
            "id": "changed",
            "variant": "hybrid",
            "input": original,
            "output": 'The town approved 21 homes. The mayor said, "We will inspect them."',
        },
        {
            "id": "copied",
            "variant": "exemplars_only",
            "input": "Describe the station at sunset.",
            "output": copied,
            "exemplars": [copied],
        },
        {"id": "missing", "variant": "qualitative", "input": "Rain arrived.", "output": "The rain came."},
    ]
    result = evaluate_writing(cases)
    rows = {row["id"]: row for row in result["records"]}
    assert rows["faithful"]["channels"]["preservation_flags"]["value"] == 1
    assert rows["changed"]["channels"]["preservation_flags"]["value"] == 0
    assert rows["changed"]["channels"]["preservation_flags"]["details"]["missing_numbers"] == ["12"]
    assert rows["changed"]["channels"]["preservation_flags"]["details"]["added_numbers"] == ["21"]
    assert rows["copied"]["channels"]["copying"]["value"] == 1
    assert rows["copied"]["channels"]["copying"]["details"]["longest_shared_words"] == 10
    assert rows["missing"]["channels"]["factuality"]["status"] == "unscored"
    assert rows["missing"]["channels"]["naturalness"]["status"] == "unscored"
    assert result["coverage"]["factuality"] == {"scored": 0, "unscored": 4, "errors": 0, "mean": None}
    assert result["fully_scored_samples"] == 0
    assert rows["faithful"]["input"] == original
    assert rows["faithful"]["input_hash"] == hashlib.sha256(original.encode()).hexdigest()
    assert rows["faithful"]["output"] == cases[0]["output"]
    assert rows["faithful"]["output_hash"] != rows["faithful"]["input_hash"]
    json.dumps(result, allow_nan=False)


def test_changed_or_invented_quotes_fail_preservation_flag():
    for output in ['He said, "Maybe."', 'He said, "Yes." Then she said, "No."']:
        result = evaluate_writing([{"id": "quote", "input": 'He said, "Yes."', "output": output}])
        assert result["records"][0]["channels"]["preservation_flags"]["value"] == 0


def test_supplied_human_ratings_are_labeled_and_not_auto_generated():
    result = evaluate_writing(
        [
            {
                "id": "human",
                "input": "The rain came.",
                "output": "Rain arrived.",
                "human": {"voice": 0.8, "naturalness": 0.7, "factuality": 1},
            }
        ]
    )
    channels = result["records"][0]["channels"]
    assert channels["voice"]["value"] == 0.8
    assert channels["naturalness"]["value"] == 0.7
    assert "calibration was not independently verified" in channels["voice"]["reason"]


def test_scorer_failure_coverage_is_independent_of_order():
    cases = [{"id": str(index)} for index in range(4)]

    def voice(case):
        return Score(0.8, "Independent fixture score")

    def raising(case):
        if case["id"] in {"2", "3"}:
            raise RuntimeError("grader unavailable")
        return Score(1)

    before = run_scorers(cases, {"voice": voice, "factuality": raising})
    after = run_scorers(cases, {"factuality": raising, "voice": voice})
    assert before == after
    assert before["completed_without_error"] == 2
    assert before["fully_scored_samples"] == 2
    assert before["coverage"]["voice"]["scored"] == 4
    assert before["coverage"]["factuality"]["errors"] == 2
    assert before["records"][2]["channels"]["voice"]["value"] == 0.8


def test_unscored_and_nonfinite_scores_do_not_turn_into_passes():
    result = run_scorers(
        [{"id": "one"}],
        {
            "missing": lambda case: None,
            "bad": lambda case: Score(float("nan")),
            "bad_details": lambda case: Score(1, details={"bad": float("inf")}),
        },
    )
    assert result["coverage"]["missing"]["unscored"] == 1
    assert result["coverage"]["missing"]["mean"] is None
    assert result["coverage"]["bad"]["errors"] == 1
    assert result["coverage"]["bad_details"]["errors"] == 1
    assert result["fully_scored_samples"] == 0


def test_grader_mutation_cannot_change_identity_or_other_channel_input():
    case = {"id": "original", "output": "retained"}

    def mutating(sample):
        sample["id"] = "different"
        sample["output"] = "changed"
        return Score(1)

    result = run_scorers(
        [case],
        {
            "mutating": mutating,
            "observing": lambda sample: Score(float(sample["output"] == "retained")),
        },
    )
    assert result["records"][0]["id"] == "original"
    assert result["records"][0]["channels"]["observing"]["value"] == 1
    assert case == {"id": "original", "output": "retained"}


@pytest.mark.parametrize("ids", [["duplicate", "duplicate"], [1, "1"], [""]])
def test_duplicate_or_invalid_sample_ids_fail_before_scoring(ids):
    with pytest.raises(ValueError, match="IDs"):
        run_scorers([{"id": value} for value in ids], {"score": lambda case: Score(1)})

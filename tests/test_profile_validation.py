"""One-author held-out checks using the real English measurement pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Literal

import pytest

from colophon.models.document import Document
from colophon.voice.manifest import LoadedDocument
from colophon.voice.models import CorpusEntry, TraitCorrection, VoiceTrait
from colophon.voice.profile import build_voice_profile
from colophon.voice.profile_validation import evaluate_profile


def source(identifier, text, *, split: Literal["train", "dev", "test"] = "train", metadata=None, work_id=None):
    entry = CorpusEntry(
        id=identifier,
        path=f"{identifier}.txt",
        author="Writer",
        work_id=work_id or identifier,
        split=split,
        genre="essay",
        topic=identifier,
        date="2024-05-01",
    )
    document = Document.from_text(text, entry.path, metadata=metadata or {})
    digest = hashlib.sha256(text.encode()).hexdigest()
    return LoadedDocument(entry, document, digest, digest, document, 0, [])


@pytest.fixture(scope="module")
def material():
    loaded = [
        source("a", "Why wait?\n\nI left because you stayed.\n\nEnough."),
        source("b", "Can we go?\n\nPerhaps I should stay, but you disagree.\n\nTomorrow."),
        source("heldout", "Why stop now?\n\nI watched because you asked.\n\nHowever, we waited.", split="test"),
    ]
    return build_voice_profile(loaded, "Writer"), loaded


def test_single_author_stability_has_real_measurements_and_no_quality_pass(material):
    profile, loaded = material
    result = evaluate_profile(profile, loaded)
    assert result["evaluation"] == "single_author_held_out_stability"
    assert result["measurement_coverage"] == {"total": 1, "measured": 1, "errors": 0}
    assert [case["id"] for case in result["cases"]] == ["heldout"]
    assert result["cases"][0]["era"] == "2020s"
    assert result["cases"][0]["measurement_config"]["model_version"]
    feature = result["features"]["sentences.mean_length"]
    assert feature["coverage"] == {"total": 1, "scored": 1, "unscored": 0, "errors": 0}
    assert feature["training"] == profile.numerical["sentences.mean_length"].model_dump()
    opening = next(trait for trait in result["traits"] if trait["id"] == "opening-question")
    assert opening["observed_rate"] == 1
    assert opening["supporting_documents"][0]["document_id"] == "heldout"
    assert "pass" not in result and "accuracy" not in result
    assert "does not establish" in result["limitations"][0]
    json.dumps(result, allow_nan=False)


def test_changing_test_cannot_change_saved_training_fit(material):
    profile, loaded = material
    before_profile = profile.model_dump_json()
    before = evaluate_profile(profile, loaded)
    changed = loaded[:2] + [
        source("heldout", "A long river crosses five counties.\n\nThe ferry returned.", split="test")
    ]
    after = evaluate_profile(profile, changed)
    assert before["training_fit_hash"] == after["training_fit_hash"]
    assert before["training_fit"] == after["training_fit"]
    assert profile.model_dump_json() == before_profile
    assert before["cases"][0]["values"] != after["cases"][0]["values"]


@pytest.mark.parametrize("kind", ["work", "text", "hash", "author_view", "ownership", "metadata"])
def test_source_and_training_identity_mismatches_reject(material, kind):
    profile, loaded = material
    loaded = list(loaded)
    if kind == "work":
        loaded[-1] = replace(loaded[-1], entry=loaded[-1].entry.model_copy(update={"work_id": "a"}))
    elif kind == "text":
        loaded[-1] = source("heldout", loaded[0].document.text, split="test")
    elif kind == "hash":
        loaded[0] = replace(loaded[0], normalized_hash="forged")
    elif kind == "author_view":
        changed = loaded[0].author_document.model_copy(update={"text": "Changed author view."})
        loaded[0] = replace(loaded[0], author_document=changed)
    elif kind == "ownership":
        loaded[0] = replace(loaded[0], entry=loaded[0].entry.model_copy(update={"genre": "changed"}))
    else:
        loaded[0] = replace(
            loaded[0], document=loaded[0].document.model_copy(update={"metadata": {"extractor": "changed"}})
        )
    with pytest.raises(ValueError):
        evaluate_profile(profile, loaded)


def test_missing_paragraph_evidence_is_unscored_not_zero(material):
    profile, loaded = material
    target = loaded[-1]
    changed = target.author_document.model_copy(update={"metadata": {"paragraph_quality": "unverified"}})
    result = evaluate_profile(profile, [*loaded[:2], replace(target, author_document=changed)])
    feature = result["features"]["paragraphs.mean_length"]
    assert feature["coverage"]["unscored"] == 1
    assert feature["heldout"]["mean"] is None
    assert result["features"]["sentences.mean_length"]["coverage"]["scored"] == 1
    opening = next(trait for trait in result["traits"] if trait["id"] == "opening-question")
    assert opening["status"] == "unscored"
    assert opening["observed_rate"] is None


def test_zero_training_variance_does_not_manufacture_a_standardized_score(material):
    profile, loaded = material
    result = evaluate_profile(profile, loaded)
    feature = result["features"]["function_words.the"]
    assert feature["training"]["stdev"] == 0
    assert feature["heldout"]["mean_shift_in_training_stdevs"] is None
    assert "zero" in feature["heldout"]["standardized_shift_reason"]


def test_interpretations_and_writer_replacements_require_semantic_review(material):
    profile, loaded = material
    annotation = VoiceTrait(
        id="interpretation",
        category="argument",
        statement="A reader's interpretation.",
        method="annotation",
        evidence=[profile.exemplars[0]],
    )
    annotated = build_voice_profile(loaded, "Writer", annotations=[annotation])
    result = evaluate_profile(annotated, loaded)
    row = next(trait for trait in result["traits"] if trait["id"] == "interpretation")
    assert row["status"] == "requires_semantic_review"
    assert "observed_rate" not in row
    corrected = build_voice_profile(
        loaded,
        "Writer",
        corrections=[
            TraitCorrection(
                trait_id="first-person",
                action="replace",
                statement="Use first person when the brief supports it.",
            )
        ],
    )
    row = next(trait for trait in evaluate_profile(corrected, loaded)["traits"] if trait["id"] == "first-person")
    assert row["training_status"] == "writer_supplied"
    assert row["status"] == "requires_semantic_review"


def test_changed_training_block_structure_is_stale_even_when_text_matches(material):
    profile, loaded = material
    original = loaded[0]
    block = original.document.blocks[0].model_copy(update={"kind": "heading"})
    changed = original.document.model_copy(update={"blocks": [block, *original.document.blocks[1:]]})
    with pytest.raises(ValueError, match="structure or extraction identity"):
        evaluate_profile(profile, [replace(original, document=changed), *loaded[1:]])


def test_failed_or_incompatible_model_measurement_stays_visible(material, monkeypatch):
    from colophon.analysis import pipeline

    profile, loaded = material
    actual = pipeline.analyze

    def changed(*args, **kwargs):
        return actual(*args, **kwargs).model_copy(update={"model_version": "different"})

    monkeypatch.setattr(pipeline, "analyze", changed)
    result = evaluate_profile(profile, loaded)
    assert result["measurement_coverage"]["errors"] == 1
    assert "configuration differs" in result["cases"][0]["reason"]
    feature = result["features"]["sentences.mean_length"]
    assert feature["coverage"] == {"total": 1, "scored": 0, "unscored": 0, "errors": 1}
    assert feature["heldout"]["mean"] is None


def test_missing_or_wrong_language_holdout_rejects(material):
    profile, loaded = material
    with pytest.raises(ValueError, match="No held-out"):
        evaluate_profile(profile, loaded[:2])
    with pytest.raises(ValueError, match="held-out"):
        evaluate_profile(profile, loaded, split="train")
    wrong = replace(loaded[-1], entry=loaded[-1].entry.model_copy(update={"language": "fr"}))
    with pytest.raises(ValueError, match="incompatible languages"):
        evaluate_profile(profile, [*loaded[:2], wrong])

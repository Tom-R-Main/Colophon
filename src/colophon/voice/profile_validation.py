"""One-author held-out stability checks, without an authorship or quality score.

The saved training profile is the fixed reference. Held-out documents can expose
variation and missing evidence; they never change the fit or certify literary
recognizability. Interpretive annotations need a separate semantic review.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from statistics import mean, median, stdev
from typing import Any

from colophon.models.features import StyleProfile
from colophon.voice.manifest import LoadedDocument, validate_loaded
from colophon.voice.models import VoiceProfile
from colophon.voice.profile import validate_evidence
from colophon.voice.rhetoric import author_evidence, extract_traits, rhetoric_features

_CONFIG_FIELDS = (
    "schema_version",
    "analyzer_version",
    "normalization_version",
    "language",
    "spacy_model",
    "spacy_version",
    "model_version",
    "vector_schema",
)
_RULE_FEATURES = {
    "opening-question": "opening_question",
    "ending-question": "ending_question",
    "short-ending": "short_ending",
    "short-paragraph": "short_paragraph",
    "contrast-markers": "contrast_markers",
    "cause-markers": "cause_markers",
    "first-person": "first_person",
    "reader-address": "reader_address",
    "hedge-markers": "hedge_markers",
    "transition-openers": "transition_openers",
}
_STRUCTURAL = {"opening_question", "ending_question", "short_ending", "short_paragraph", "transition_openers"}


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash(value: object) -> str:
    return _text_hash(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def _config(measurement: StyleProfile) -> dict[str, Any]:
    return {field: getattr(measurement, field) for field in _CONFIG_FIELDS}


def _metadata(item: LoadedDocument) -> dict[str, Any]:
    date = item.entry.date
    year_match = re.search(r"\b(\d{4})\b", date or "")
    era = f"{int(year_match.group(1)) // 10 * 10}s" if year_match else "unknown"
    return {
        "id": item.entry.id,
        "work_id": item.entry.work_id,
        "author": item.entry.effective_author,
        "language": item.entry.language,
        "genre": item.entry.genre,
        "topic": item.entry.topic,
        "date": date,
        "era": era,
        "source_hash": item.source_hash,
        "normalized_hash": item.normalized_hash,
        "author_text_hash": _text_hash(item.author_document.text),
        "extraction_metadata": item.document.metadata,
    }


def _bind_training(profile: VoiceProfile, loaded: list[LoadedDocument]) -> tuple[list[LoadedDocument], dict[str, Any]]:
    if profile.schema_version != "1" or profile.extractor_version != "1" or not profile.documents:
        raise ValueError("Unsupported or empty profile; re-extract before held-out validation")
    train = sorted(
        (item for item in loaded if item.entry.split == "train" and item.entry.effective_author == profile.author),
        key=lambda item: item.entry.id,
    )
    saved = {document.entry.id: document for document in profile.documents}
    if len(saved) != len(profile.documents) or set(saved) != {item.entry.id for item in train}:
        raise ValueError("Profile training document IDs differ from the included training corpus")
    configurations = []
    for item in train:
        document = saved[item.entry.id]
        actual = _text_hash(item.document.text)
        if actual != item.normalized_hash or actual != document.normalized_hash:
            raise ValueError(f"Stale or forged normalized training source: {item.entry.id}")
        if item.source_hash != document.source_hash:
            raise ValueError(f"Training source identity changed: {item.entry.id}")
        if item.entry.model_dump(mode="json") != document.entry.model_dump(mode="json"):
            raise ValueError(f"Training ownership, grouping or exclusions changed: {item.entry.id}")
        if item.document.metadata != document.extraction_metadata:
            raise ValueError(f"Training extraction metadata changed: {item.entry.id}")
        if document.measurement.source_hash != _text_hash(item.author_document.text):
            raise ValueError(f"Training author-view text changed: {item.entry.id}")
        configuration = _config(document.measurement)
        if configuration["schema_version"] != "2" or configuration["analyzer_version"] != "2":
            raise ValueError("Legacy training measurements require re-extraction")
        if item.entry.language != profile.language or configuration["language"] != profile.language:
            raise ValueError("Training profile and source languages are incompatible")
        configurations.append(configuration)
    if not train:
        raise ValueError("No included training sources for this profile")
    if any(configuration != configurations[0] for configuration in configurations[1:]):
        raise ValueError("Training documents have incompatible measurement/model configurations")
    # Bind source structure as well as text. Equal text can have different block
    # ownership or paragraph ranges, which changes both measurement and traits.
    identity = [
        {
            "entry": item.entry.model_dump(mode="json"),
            "source_hash": item.source_hash,
            "normalized_hash": item.normalized_hash,
            "author_text_hash": _text_hash(item.author_document.text),
            "blocks": [block.model_dump(mode="json") for block in item.document.blocks],
            "extraction_metadata": item.document.metadata,
        }
        for item in train
    ]
    if _hash(identity) != profile.corpus_hash:
        raise ValueError("Training corpus structure or extraction identity differs from the saved profile")
    if _hash({"corpus": profile.corpus_hash, "config": profile.config_hash}) != profile.profile_id:
        raise ValueError("Profile identity does not match its corpus/configuration hashes")
    for trait in profile.traits:
        for span in trait.evidence + trait.counterexamples:
            validate_evidence(span, train)
    for span in profile.exemplars:
        validate_evidence(span, train)
    return train, configurations[0]


def _feature(measurement: StyleProfile, path: str) -> float | None:
    group, field = path.split(".", 1)
    if measurement.availability.get(group) != "measured":
        return None
    value = getattr(measurement, group, None)
    if value is None:
        return None
    value = value.frequencies.get(field) if group == "function_words" else getattr(value, field, None)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _features(profile: VoiceProfile, cases: list[dict[str, Any]]) -> dict[str, Any]:
    report = {}
    for path, reference in sorted(profile.numerical.items()):
        if not all(
            math.isfinite(value) for value in reference.model_dump().values() if isinstance(value, (int, float))
        ):
            raise ValueError(f"Training feature summary is not finite: {path}")
        scored = [
            case["values"][path]
            for case in cases
            if case["status"] == "measured" and case["values"].get(path) is not None
        ]
        errors = sum(case["status"] == "error" for case in cases)
        unscored = len(cases) - errors - len(scored)
        average = mean(scored) if scored else None
        shift = (average - reference.mean) / reference.stdev if average is not None and reference.stdev > 0 else None
        if not scored:
            shift_reason = "No compatible held-out measurements"
        elif reference.stdev == 0:
            shift_reason = "Training standard deviation is zero; standardized shift is unavailable"
        else:
            shift_reason = None
        in_range = sum(reference.minimum <= value <= reference.maximum for value in scored)
        report[path] = {
            "unit": reference.unit,
            "training": reference.model_dump(mode="json"),
            "coverage": {"total": len(cases), "scored": len(scored), "unscored": unscored, "errors": errors},
            "heldout": {
                "mean": average,
                "median": median(scored) if scored else None,
                "minimum": min(scored) if scored else None,
                "maximum": max(scored) if scored else None,
                "stdev": stdev(scored) if len(scored) > 1 else 0.0 if scored else None,
                "within_training_range": in_range if scored else None,
                "within_training_range_rate": in_range / len(scored) if scored else None,
                "mean_shift_in_training_stdevs": shift,
                "standardized_shift_reason": shift_reason,
            },
        }
    return report


def _trait_checks(
    profile: VoiceProfile, train: list[LoadedDocument], heldout: list[LoadedDocument]
) -> list[dict[str, Any]]:
    original_rules = {trait.id for trait in extract_traits(train)}
    observed = {item.entry.id: {trait.id: trait for trait in extract_traits([item])} for item in heldout}
    feature_rows = {}
    for item in heldout:
        row = rhetoric_features(
            item.author_document.text, item.entry.language, source_text=item.document.text, blocks=item.document.blocks
        )
        if item.author_document.metadata.get("paragraph_quality") == "unverified":
            row = {name: value for name, value in row.items() if name not in _STRUCTURAL}
        feature_rows[item.entry.id] = row
    results = []
    for trait in profile.traits:
        record: dict[str, Any] = {
            "id": trait.id,
            "category": trait.category,
            "training_statement": trait.statement,
            "training_status": trait.status,
            "method": trait.method,
        }
        if trait.status == "rejected":
            record.update(status="unscored", reason="The trait was rejected in the saved profile")
        elif trait.method != "rule" or trait.status == "writer_supplied" or trait.id not in original_rules:
            record.update(
                status="requires_semantic_review",
                reason="Exact training citations do not automatically validate an interpretation on unseen writing",
            )
        else:
            support, counterexamples, unavailable = [], [], []
            for item in heldout:
                doc_id = item.entry.id
                if trait.id == "quote-adjacency":
                    eligible = bool(author_evidence(item))
                else:
                    key = _RULE_FEATURES.get(trait.id)
                    eligible = key is not None and key in feature_rows[doc_id]
                if not eligible:
                    unavailable.append(doc_id)
                elif trait.id in observed[doc_id]:
                    match = observed[doc_id][trait.id]
                    support.append(
                        {"document_id": doc_id, "evidence": [span.model_dump(mode="json") for span in match.evidence]}
                    )
                else:
                    counterexamples.append(doc_id)
            eligible_count = len(support) + len(counterexamples)
            record.update(
                status="observed" if eligible_count else "unscored",
                coverage={"total": len(heldout), "eligible": eligible_count, "unscored": len(unavailable)},
                supporting_documents=support,
                counterexample_documents=counterexamples,
                unavailable_documents=unavailable,
                observed_rate=len(support) / eligible_count if eligible_count else None,
                scope="The original literal rule was checked; "
                "its rhetorical intention or distinctiveness was not scored",
            )
        results.append(record)
    return results


def evaluate_profile(profile: VoiceProfile, loaded: list[LoadedDocument], *, split: str = "test") -> dict[str, Any]:
    """Compare a fixed one-author profile with independent unseen pieces.

    This reports feature variation and literal-rule recurrence. There is no
    aggregate pass, authorship confidence, generated output or human voice score.
    """
    from colophon.analysis.pipeline import analyze

    if split not in {"dev", "test"}:
        raise ValueError("Profile validation requires a held-out dev or test split")
    active = [item for item in loaded if item.entry.include]
    validate_loaded(active)
    for item in active:
        if _text_hash(item.document.text) != item.normalized_hash:
            raise ValueError(f"Stale or forged normalized source hash: {item.entry.id}")
    train, configuration = _bind_training(profile, active)
    heldout = sorted(
        (item for item in active if item.entry.split == split and item.entry.effective_author == profile.author),
        key=lambda item: item.entry.id,
    )
    if not heldout:
        raise ValueError(f"No held-out {split} documents for author {profile.author!r}")
    if any(item.entry.language != profile.language for item in heldout):
        raise ValueError("Held-out sources and profile have incompatible languages")
    training_work_ids = {document.entry.work_id for document in profile.documents}
    if any(item.entry.work_id in training_work_ids for item in heldout):
        raise ValueError("A held-out work occurs in the saved training profile")
    cases = []
    for item in heldout:
        record = _metadata(item)
        try:
            measurement = analyze(
                item.author_document,
                lang=item.entry.language,
                spacy_model=configuration["spacy_model"] or "en_core_web_sm",
            )
            actual_config = _config(measurement)
            if actual_config != configuration:
                raise ValueError("Held-out measurement/model configuration differs from the fixed training profile")
            record.update(
                status="measured",
                measurement_config=actual_config,
                measurement_availability=measurement.availability,
                warnings=measurement.warnings,
                values={path: _feature(measurement, path) for path in profile.numerical},
            )
        except Exception as error:
            record.update(status="error", reason=f"{type(error).__name__}: {error}", values={})
        cases.append(record)
    traits = _trait_checks(profile, train, heldout)
    fit = {
        "profile_id": profile.profile_id,
        "corpus_hash": profile.corpus_hash,
        "config_hash": profile.config_hash,
        "training_documents": [
            {
                "id": doc.entry.id,
                "work_id": doc.entry.work_id,
                "source_hash": doc.source_hash,
                "normalized_hash": doc.normalized_hash,
            }
            for doc in profile.documents
        ],
        "numerical": {key: value.model_dump(mode="json") for key, value in sorted(profile.numerical.items())},
        "measurement_config": configuration,
    }
    result = {
        "schema_version": "1",
        "evaluation": "single_author_held_out_stability",
        "split": split,
        "author": profile.author,
        "language": profile.language,
        "profile_id": profile.profile_id,
        "training_fit_hash": _hash(fit),
        "training_fit": fit,
        "cases": cases,
        "measurement_coverage": {
            "total": len(cases),
            "measured": sum(case["status"] == "measured" for case in cases),
            "errors": sum(case["status"] == "error" for case in cases),
        },
        "features": _features(profile, cases),
        "traits": traits,
        "trait_coverage": {
            "literal_rules_checked": sum(row["status"] == "observed" for row in traits),
            "requires_semantic_review": sum(row["status"] == "requires_semantic_review" for row in traits),
            "unscored": sum(row["status"] == "unscored" for row in traits),
        },
        "limitations": [
            "Same-author feature stability does not establish authorship discrimination "
            "or recognizable literary voice.",
            "Training ranges are descriptive; an out-of-range held-out value is not an automatic failure.",
            "No generated writing, provider call, blinded human rating or aggregate quality pass is produced.",
            "Interpretive annotations require separate semantic review on unseen source passages.",
            "Source hashes bind the saved profile to supplied local artifacts; they are not authenticity signatures.",
        ],
    }
    json.dumps(result, allow_nan=False)
    return result

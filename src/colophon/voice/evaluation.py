"""Offline evaluation with training-only fitting and independently visible scores.

The retrieval methods are transparent baselines, not generative model comparisons.
Writing evaluation accepts already-produced outputs and never calls a provider.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from statistics import mean, pstdev
from typing import TYPE_CHECKING, Any, Literal

from colophon.lang import get_profile

if TYPE_CHECKING:
    from colophon.voice.manifest import LoadedDocument

METHODS = ("numerical", "exemplars_only", "qualitative", "hybrid")
STRUCTURAL_FEATURES = {"short_paragraph", "opening_question", "ending_question", "short_ending", "transition_openers"}
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
_NUMBER = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?:%)?")
_QUOTES = re.compile(r'"([^"\n]+)"|“([^”]+)”|«([^»]+)»')


class UnavailableMeasurement(ValueError):
    """The requested channel has insufficient compatible evidence."""


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def _validate_corpus(loaded: list[LoadedDocument], split: str) -> list[LoadedDocument]:
    from colophon.voice.manifest import validate_loaded

    if split not in {"dev", "test"}:
        raise ValueError("Evaluate only a held-out dev or test split")
    active = [item for item in loaded if item.entry.include]
    validate_loaded(active)
    if not any(item.entry.split == "train" for item in active):
        raise ValueError("Evaluation requires training documents")
    if not any(item.entry.split == split for item in active):
        raise ValueError(f"Evaluation requires held-out {split} documents")
    if len({item.entry.language for item in active}) != 1:
        raise ValueError("Evaluate one compatible language at a time")
    return active


def _numeric_features(item: LoadedDocument) -> dict[str, float]:
    from colophon.analysis.pipeline import analyze

    profile = analyze(
        item.author_document,
        analyzers=["sentences", "function_words", "punctuation", "paragraphs", "contractions"],
        lang=item.entry.language,
    )
    values: dict[str, float] = {}
    for group, names in (
        ("sentences", ("mean_length", "median_length", "stdev_length", "skewness")),
        ("punctuation", ("comma", "period", "semicolon", "colon", "dash", "exclamation", "question")),
        ("paragraphs", ("mean_length", "median_length", "one_sentence_ratio")),
        ("contractions", ("rate_per_1000",)),
    ):
        measurement = getattr(profile, group)
        if measurement is not None:
            values.update({f"{group}.{name}": float(getattr(measurement, name)) for name in names})
    if profile.function_words:
        # Fixed canonical coordinate names; values missing from an old truncated
        # analyzer cannot shift the coordinates of other function words.
        for word in get_profile(item.entry.language).function_words:
            values[f"function_words.{word}"] = profile.function_words.frequencies.get(word, 0.0)
    return values


def _fit_scaler(rows: list[dict[str, float]]) -> dict[str, Any]:
    keys = sorted(set.intersection(*(set(row) for row in rows))) if rows else []
    means = {key: mean(row[key] for row in rows) for key in keys}
    scales = {key: pstdev(row[key] for row in rows) for key in keys}
    keys = [key for key in keys if scales[key] > 1e-12]
    return {"features": keys, "means": {key: means[key] for key in keys}, "scales": {key: scales[key] for key in keys}}


def _transform(row: dict[str, float], scaler: dict[str, Any]) -> list[float]:
    missing = sorted(set(scaler["features"]) - set(row))
    if missing:
        raise UnavailableMeasurement(f"Query lacks available measurements for training coordinates: {missing}")
    return [(row[key] - scaler["means"][key]) / scaler["scales"][key] for key in scaler["features"]]


def _distance(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise UnavailableMeasurement("No compatible varying training features")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(left, right)) / len(left))


def _char_counts(text: str) -> Counter[str]:
    cleaned = " ".join(text.casefold().split())
    return Counter(cleaned[i : i + 3] for i in range(len(cleaned) - 2))


def _char_vector(text: str, vocabulary: list[str]) -> list[float]:
    counts = _char_counts(text)
    vector = [float(counts.get(key, 0)) for key in vocabulary]
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def _cosine_distance(left: list[float], right: list[float]) -> float:
    if not any(left) or not any(right):
        raise UnavailableMeasurement("Text has no features in the training character vocabulary")
    return max(0.0, min(2.0, 1 - sum(x * y for x, y in zip(left, right))))


def _centroids(train: list[LoadedDocument], vectors: list[list[float]]) -> dict[str, list[float]]:
    groups: dict[str, list[list[float]]] = defaultdict(list)
    for item, vector in zip(train, vectors):
        groups[item.entry.effective_author].append(vector)
    return {author: [mean(values) for values in zip(*rows)] for author, rows in sorted(groups.items())}


def _fit_threshold(dev: list[tuple[float, bool]]) -> dict[str, Any]:
    if not dev or {known for _, known in dev} != {True, False}:
        return {"status": "unavailable", "reason": "Requires dev examples of both known and unknown authors"}
    distances = sorted(set(distance for distance, _ in dev))
    candidates = [distances[0] - 1e-9, *distances, distances[-1] + 1e-9]

    def error(threshold: float) -> float:
        false_reject = mean(float(distance > threshold) for distance, known in dev if known)
        false_accept = mean(float(distance <= threshold) for distance, known in dev if not known)
        return (false_reject + false_accept) / 2

    threshold = min(candidates, key=lambda candidate: (error(candidate), candidate))
    return {
        "status": "calibrated_on_dev",
        "distance_threshold": threshold,
        "balanced_error_on_dev": error(threshold),
        "dev_samples": len(dev),
    }


def evaluate_corpus(loaded: list[LoadedDocument], *, split: str = "test") -> dict[str, Any]:
    """Compare training-only retrieval baselines on independent held-out writing.

    ``qualitative`` is a rule-marker vector, ``exemplars_only`` is nearest
    training-document character-trigram retrieval. Neither means LLM generation.
    Unknown-author acceptance thresholds, when available, are fitted on dev only.
    """
    from colophon.voice.rhetoric import rhetoric_features

    active = _validate_corpus(loaded, split)
    language = active[0].entry.language
    train = sorted((item for item in active if item.entry.split == "train"), key=lambda item: item.entry.id)
    targets = sorted((item for item in active if item.entry.split == split), key=lambda item: item.entry.id)
    dev = sorted((item for item in active if item.entry.split == "dev"), key=lambda item: item.entry.id)
    authors = sorted({item.entry.effective_author for item in train})
    if len(authors) < 2:
        raise ValueError("Authorship comparison requires at least two training authors")
    methods: dict[str, dict[str, Any]] = {}
    method_errors: dict[str, str] = {}
    numeric: dict[str, dict[str, float]] = {}
    for item in active:
        try:
            numeric[item.entry.id] = _numeric_features(item)
        except Exception as exc:
            method_errors[item.entry.id] = f"{type(exc).__name__}: {exc}"
    qualitative = {}
    for item in active:
        features = rhetoric_features(
            item.author_document.text,
            language,
            source_text=item.document.text,
            blocks=item.document.blocks,
        )
        if item.author_document.metadata.get("paragraph_quality") == "unverified":
            features = {key: value for key, value in features.items() if key not in STRUCTURAL_FEATURES}
        qualitative[item.entry.id] = features
    scaler_n = (
        _fit_scaler([numeric[item.entry.id] for item in train])
        if all(item.entry.id in numeric for item in train)
        else None
    )
    scaler_q = _fit_scaler([qualitative[item.entry.id] for item in train])
    counts: Counter[str] = Counter()
    for item in train:
        counts.update(_char_counts(item.author_document.text))
    vocabulary = [key for key, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:512]]
    train_c = [_char_vector(item.author_document.text, vocabulary) for item in train]
    references: dict[str, dict[str, list[float]]] = {}
    if scaler_n and scaler_n["features"]:
        references["numerical"] = _centroids(train, [_transform(numeric[item.entry.id], scaler_n) for item in train])
    if scaler_q["features"]:
        references["qualitative"] = _centroids(
            train, [_transform(qualitative[item.entry.id], scaler_q) for item in train]
        )

    def distances(item: LoadedDocument, method: str) -> dict[str, float]:
        if method == "exemplars_only":
            query = _char_vector(item.author_document.text, vocabulary)
            return {
                author: min(
                    _cosine_distance(query, vector)
                    for reference, vector in zip(train, train_c)
                    if reference.entry.effective_author == author
                )
                for author in authors
            }
        if method == "hybrid":
            component = [distances(item, name) for name in ("numerical", "qualitative")]
            return {author: mean(channel[author] for channel in component) for author in authors}
        if method not in references:
            raise UnavailableMeasurement("Training measurements unavailable or have no varying features")
        row = numeric[item.entry.id] if method == "numerical" else qualitative[item.entry.id]
        scaler = scaler_n if method == "numerical" else scaler_q
        assert scaler is not None
        query = _transform(row, scaler)
        return {author: _distance(query, reference) for author, reference in references[method].items()}

    for method in METHODS:
        # When evaluating dev itself, no dev-fitted threshold is applied back to
        # those same samples. Test evaluation may use a separately fixed dev set.
        dev_distances: list[tuple[float, bool]] = []
        dev_errors = []
        if split == "test":
            for item in dev:
                try:
                    dev_distances.append(
                        (min(distances(item, method).values()), item.entry.effective_author in authors)
                    )
                except Exception as exc:
                    dev_errors.append({"id": item.entry.id, "reason": f"{type(exc).__name__}: {exc}"})
                    continue
        threshold = _fit_threshold(dev_distances)
        threshold.update(
            {
                "dev_total": len(dev) if split == "test" else 0,
                "dev_scored": len(dev_distances),
                "dev_errors": dev_errors,
            }
        )
        cases = []
        for item in targets:
            entry = item.entry
            case: dict[str, Any] = {
                "id": entry.id,
                "author": entry.effective_author,
                "work_id": entry.work_id,
                "topic": entry.topic,
                "genre": entry.genre,
                "known_author": entry.effective_author in authors,
                "source_hash": item.normalized_hash,
            }
            try:
                ranked = sorted(distances(item, method).items(), key=lambda pair: (pair[1], pair[0]))
                best_author, best_distance = ranked[0]
                tied = [author for author, distance in ranked if math.isclose(distance, best_distance, abs_tol=1e-12)]
                predicted = best_author if len(tied) == 1 else None
                rejected = (
                    best_distance > threshold["distance_threshold"] if "distance_threshold" in threshold else None
                )
                case.update(
                    {
                        "status": "scored",
                        "ranking": [{"author": author, "distance": distance} for author, distance in ranked],
                        "predicted_author": None if rejected else predicted,
                        "tied_authors": tied if len(tied) > 1 else [],
                        "rejected_unknown": rejected,
                        "correct_known_author": predicted == entry.effective_author and not rejected
                        if case["known_author"]
                        else None,
                    }
                )
            except UnavailableMeasurement as exc:
                case.update({"status": "unscored", "reason": str(exc)})
            except Exception as exc:
                case.update({"status": "error", "reason": f"{type(exc).__name__}: {exc}"})
            cases.append(case)
        scored = [case for case in cases if case["status"] == "scored"]
        known = [case for case in scored if case["known_author"]]
        unknown = [case for case in scored if not case["known_author"]]
        per_author = {}
        for author in sorted({case["author"] for case in known}):
            rows = [case for case in known if case["author"] == author]
            per_author[author] = {
                "samples": len(rows),
                "accuracy": mean(float(case["correct_known_author"]) for case in rows),
            }
        methods[method] = {
            "cases": cases,
            "coverage": {
                "total": len(cases),
                "scored": len(scored),
                "unscored": sum(case["status"] == "unscored" for case in cases),
                "errors": sum(case["status"] == "error" for case in cases),
            },
            "known_author_accuracy": mean(float(case["correct_known_author"]) for case in known) if known else None,
            "per_author": per_author,
            "macro_author_accuracy": mean(row["accuracy"] for row in per_author.values()) if per_author else None,
            "unknown_author_calibration": threshold,
            "unknown_author_rejection_rate": mean(float(case["rejected_unknown"]) for case in unknown)
            if unknown and "distance_threshold" in threshold
            else None,
        }
    fit = {
        "train": [
            {
                "id": item.entry.id,
                "author": item.entry.effective_author,
                "text_hash": _text_hash(item.author_document.text),
            }
            for item in train
        ],
        "numerical_scaler": scaler_n,
        "qualitative_scaler": scaler_q,
        "character_trigrams": vocabulary,
    }
    result = {
        "schema_version": "1",
        "evaluation": "held_out_retrieval",
        "split": split,
        "language": language,
        "training_fit_hash": _hash(fit),
        "training_fit": fit,
        "methods": methods,
        "measurement_errors": method_errors,
        "method_definitions": {
            "numerical": (
                "Equal-document author centroids of measured features; varying coordinates z-scaled on training only"
            ),
            "exemplars_only": (
                "Nearest training-document character-trigram cosine distance; a topic-sensitive retrieval proxy"
            ),
            "qualitative": "Equal-document author centroids of transparent rule-marker features, not LLM judgments",
            "hybrid": "Mean of numerical and rule-marker standardized RMS distances; fixed weights, not tuned on test",
        },
        "limitations": [
            "These retrieval baselines do not demonstrate generated-writing voice fidelity or factual preservation.",
            (
                "Distances are not confidence probabilities; unknown-author rejection is unavailable "
                "without known and unknown dev authors."
            ),
            (
                "No confidence interval is reported; assess uncertainty using independent works and more authors, "
                "not correlated chunks."
            ),
            (
                "Character trigrams and explicit markers can encode topic/register; "
                "inspect cross-topic cases and human judgments."
            ),
        ],
    }
    json.dumps(result, allow_nan=False)
    return result


@dataclass(frozen=True)
class Score:
    """A named scorer's result; null is unavailable, never an implicit pass."""

    value: float | None = None
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    direction: Literal["higher", "lower"] = "higher"


def run_scorers(
    cases: list[dict[str, Any]], scorers: Mapping[str, Callable[[dict[str, Any]], Score | None]]
) -> dict[str, Any]:
    """Run every independent channel, preserving errors regardless of order."""
    ids = [case.get("id") for case in cases]
    if any(not isinstance(case_id, str) or not case_id.strip() for case_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Evaluation sample IDs must be unique nonempty strings")
    if not scorers or any(not isinstance(name, str) or not name.strip() for name in scorers):
        raise ValueError("At least one named scorer is required")
    records = []
    for case in cases:
        channels = {}
        for name, scorer in scorers.items():
            try:
                # A failed or stateful grader cannot alter another channel's
                # input or the original sample identity.
                score = scorer(deepcopy(case))
                if score is None:
                    score = Score(reason="Scorer returned no measurement")
                if not isinstance(score, Score):
                    raise TypeError("Scorers must return Score or None")
                if score.direction not in {"higher", "lower"}:
                    raise ValueError("Score direction must be higher or lower")
                if score.value is not None and (isinstance(score.value, bool) or not math.isfinite(score.value)):
                    raise ValueError("Score must be finite or unavailable")
                value = {
                    "status": "scored" if score.value is not None else "unscored",
                    "value": score.value,
                    "reason": score.reason,
                    "details": score.details,
                    "direction": score.direction,
                }
                json.dumps(value, allow_nan=False)
                channels[name] = value
            except Exception as exc:
                channels[name] = {"status": "error", "value": None, "reason": f"{type(exc).__name__}: {exc}"}
        states = {channel["status"] for channel in channels.values()}
        records.append(
            {
                "id": case["id"],
                "variant": case.get("variant", "supplied"),
                "status": "error" if "error" in states else "partial" if "unscored" in states else "complete",
                "channels": dict(sorted(channels.items())),
            }
        )
    coverage = {}
    for name in sorted(scorers):
        channels = [record["channels"][name] for record in records]
        scored = [channel["value"] for channel in channels if channel["status"] == "scored"]
        coverage[name] = {
            "scored": len(scored),
            "unscored": sum(channel["status"] == "unscored" for channel in channels),
            "errors": sum(channel["status"] == "error" for channel in channels),
            "mean": mean(scored) if scored else None,
        }
    return {
        "records": records,
        "total_samples": len(cases),
        "completed_without_error": sum(record["status"] != "error" for record in records),
        "fully_scored_samples": sum(record["status"] == "complete" for record in records),
        "coverage": coverage,
    }


def _rating(case: dict[str, Any], name: str) -> Score:
    value = case.get("human", {}).get(name)
    if value is None:
        return Score(reason=f"No human {name} rating supplied")
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"Human {name} rating must be a finite number between 0 and 1")
    return Score(float(value), "Supplied human rating; reviewer calibration was not independently verified")


def _preservation(case: dict[str, Any]) -> Score:
    def numbers(text: str) -> Counter[str]:
        return Counter(
            number.replace(",", "") if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:%)?", number) else number
            for number in _NUMBER.findall(text)
        )

    def quotes(text: str) -> Counter[str]:
        return Counter(" ".join(next(part for part in match if part).split()) for match in _QUOTES.findall(text))

    before_numbers, after_numbers = numbers(case["input"]), numbers(case["output"])
    before_quotes, after_quotes = quotes(case["input"]), quotes(case["output"])
    details = {
        "missing_numbers": list((before_numbers - after_numbers).elements()),
        "added_numbers": list((after_numbers - before_numbers).elements()),
        "missing_quotes": list((before_quotes - after_quotes).elements()),
        "added_quotes": list((after_quotes - before_quotes).elements()),
        "scope": (
            "Exact number and direct-quotation multiset checks only; "
            "meaning, attribution and other factual claims remain unverified"
        ),
    }
    return Score(
        float(before_numbers == after_numbers and before_quotes == after_quotes),
        "Preservation flags; passing is not proof of factuality",
        details,
    )


def _copying(case: dict[str, Any]) -> Score:
    sources = case.get("exemplars", []) + case.get("references", [])
    if not sources:
        return Score(reason="No exemplar/reference corpus supplied for copying checks", direction="lower")
    if not any(_words(text) for text in sources):
        return Score(reason="No lexical source material supplied for copying checks", direction="lower")
    output_words = _words(case["output"])
    longest = 0
    passage = ""
    source_index = None
    copied_positions: set[int] = set()
    for index, text in enumerate(sources):
        source_words = _words(text)
        match = SequenceMatcher(None, output_words, source_words, autojunk=False).find_longest_match()
        if match.size > longest:
            longest, passage, source_index = match.size, " ".join(output_words[match.a : match.a + match.size]), index
        source_ngrams = {tuple(source_words[i : i + 8]) for i in range(len(source_words) - 7)}
        for i in range(len(output_words) - 7):
            if tuple(output_words[i : i + 8]) in source_ngrams:
                copied_positions.update(range(i, i + 8))
    ratio = len(copied_positions) / len(output_words) if output_words else 0.0
    return Score(
        ratio,
        "Fraction of output lexical words covered by exact shared 8-word sequences; lower means less overlap",
        {
            "longest_shared_words": longest,
            "longest_shared_passage": passage,
            "source_index": source_index,
            "sources_checked": len(sources),
            "normalization": "Casefolded lexical tokens; punctuation and numeric tokens are excluded",
            "scope": (
                "Common phrases and necessary quotations require human interpretation; "
                "paraphrased copying is not detected"
            ),
        },
        direction="lower",
    )


def _voice(case: dict[str, Any]) -> Score:
    if case.get("human", {}).get("voice") is not None:
        return _rating(case, "voice")
    references = case.get("references", [])
    if not references:
        return Score(reason="No human voice rating or author reference passages supplied")
    output = _char_counts(case["output"])
    similarities = []
    for reference in references:
        target = _char_counts(reference)
        denominator = math.sqrt(
            sum(value * value for value in output.values()) * sum(value * value for value in target.values())
        )
        if denominator:
            similarities.append(sum(value * target[key] for key, value in output.items()) / denominator)
    if not similarities:
        return Score(reason="Reference/output too short for character-trigram comparison")
    return Score(
        mean(similarities),
        "Character-trigram overlap proxy; topic-sensitive and not validated voice fidelity",
        {"metric": "mean_character_trigram_cosine", "references": len(similarities)},
    )


def evaluate_writing(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate supplied variants without generating text or spending API credits.

    Each case requires unique ``id``, ``input`` and ``output`` strings. Optional
    ``variant``, ``references``/``exemplars`` string lists and ``human`` ratings
    (voice, naturalness, factuality in [0,1]) remain separately inspectable.
    """
    if not cases:
        raise ValueError("At least one supplied writing case is required")
    for case in cases:
        if not isinstance(case.get("variant", "supplied"), str) or not case.get("variant", "supplied").strip():
            raise ValueError("variant must be a nonempty string")
        for key in ("input", "output"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ValueError(f"Writing case requires a nonempty {key} string")
        for key in ("references", "exemplars"):
            values = case.get(key, [])
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"{key} must be a list of strings")
        if not isinstance(case.get("human", {}), dict):
            raise ValueError("human ratings must be an object")
    result = run_scorers(
        cases,
        {
            "voice": _voice,
            "preservation_flags": _preservation,
            "copying": _copying,
            "naturalness": lambda case: _rating(case, "naturalness"),
            "factuality": lambda case: _rating(case, "factuality"),
        },
    )
    by_id = {case["id"]: case for case in cases}
    for record in result["records"]:
        case = by_id[record["id"]]
        record.update(
            {
                "input": case["input"],
                "output": case["output"],
                "input_hash": _text_hash(case["input"]),
                "output_hash": _text_hash(case["output"]),
                "reference_text_hashes": [_text_hash(text) for text in case.get("references", [])],
                "exemplar_text_hashes": [_text_hash(text) for text in case.get("exemplars", [])],
            }
        )
    result.update(
        {
            "schema_version": "1",
            "evaluation": "supplied_writing_outputs",
            "limitations": [
                "No outputs were generated and no provider was called.",
                "Automatic preservation checks cover numbers and quotations, not full factual equivalence.",
                (
                    "Voice overlap and copying measurements can be confounded by topic; "
                    "blinded human ratings remain necessary."
                ),
                (
                    "Unavailable ratings remain unscored, and a good channel cannot compensate "
                    "for failure in another channel."
                ),
            ],
        }
    )
    variants = sorted({record["variant"] for record in result["records"]})
    result["variants"] = {
        variant: {
            name: {"scored": len(values), "mean": mean(values) if values else None}
            for name in result["coverage"]
            for values in [
                [
                    record["channels"][name]["value"]
                    for record in result["records"]
                    if record["variant"] == variant and record["channels"][name]["status"] == "scored"
                ]
            ]
        }
        for variant in variants
    }
    json.dumps(result, allow_nan=False)
    return result

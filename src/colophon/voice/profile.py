"""Build a portable profile from independent, included training documents."""

from __future__ import annotations

import hashlib
import json
import re
from math import isfinite
from statistics import mean, median, stdev
from typing import TYPE_CHECKING

from colophon.voice.models import (
    EvidenceSpan,
    NumericSummary,
    TraitCorrection,
    VoiceDocument,
    VoiceProfile,
    VoiceTrait,
)
from colophon.voice.rhetoric import author_evidence, extract_traits

if TYPE_CHECKING:
    from colophon.voice.manifest import LoadedDocument


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def validate_evidence(span: EvidenceSpan, loaded: list[LoadedDocument]) -> None:
    """Validate identity, exact bytes and ownership before admitting a claim."""
    source = next((doc for doc in loaded if doc.entry.id == span.document_id), None)
    if source is None or source.normalized_hash != span.source_hash:
        raise ValueError(f"Unknown or stale evidence document: {span.document_id}")
    if span.end <= span.start or source.document.text[span.start : span.end] != span.text:
        raise ValueError(f"Evidence does not match source offsets: {span.document_id}")
    if source.author_document.text[span.start : span.end] != span.text:
        raise ValueError("Evidence crosses excluded or non-author material")
    blocks = getattr(source.author_document, "blocks", [])
    block = next((block for block in blocks if block.id == span.block_id), None)
    if (
        block is None
        or block.attribution != "author"
        or block.kind not in {"paragraph", "unknown"}
        or span.start < block.char_offset_start
        or span.end > block.char_offset_end
    ):
        raise ValueError("Evidence does not belong to the stated author paragraph block")


def _numeric_summaries(documents: list[VoiceDocument]) -> dict[str, NumericSummary]:
    units = {
        "readability.flesch_reading_ease": "Flesch reading-ease score",
        "readability.flesch_kincaid_grade": "grade estimate",
        "sentences.mean_length": "lexical words/sentence",
        "sentences.median_length": "lexical words/sentence",
        "sentences.stdev_length": "lexical words/sentence",
        "vocabulary.ttr": "unique types/token (sample-length dependent)",
        "vocabulary.hapax_ratio": "single-occurrence types/unique types",
        "vocabulary.yules_k": "Yule K statistic",
        "pos.adjective_noun_ratio": "adjectives/nouns",
        "pos.adverb_density": "adverbs/100 lexical words",
        "pos.verb_density": "verbs/100 lexical words",
        "contractions.rate_per_1000": "contractions/1000 lexical words",
        "sentence_openers.conjunction_start_rate": "percent of sentences",
        "paragraphs.mean_length": "lexical words/paragraph",
        "paragraphs.median_length": "lexical words/paragraph",
        "paragraphs.one_sentence_ratio": "one-sentence paragraphs/paragraphs",
        "dialogue.quoted_word_ratio": "quoted lexical words/lexical words in analyzed author text",
        "syntax.mean_tree_depth": "dependency edges",
    }
    for field in (
        "comma",
        "period",
        "semicolon",
        "colon",
        "dash",
        "exclamation",
        "question",
        "ellipsis",
        "parenthesis",
        "quotation",
    ):
        units[f"punctuation.{field}"] = "marks/1000 lexical words"
    result = {}
    for path, unit in units.items():
        group, field = path.split(".")
        values = []
        for doc in documents:
            feature = getattr(doc.measurement, group)
            value = getattr(feature, field) if feature is not None else None
            if value is not None and isfinite(value):
                values.append(float(value))
        if values:
            result[path] = NumericSummary(
                mean=mean(values),
                median=median(values),
                minimum=min(values),
                maximum=max(values),
                stdev=stdev(values) if len(values) > 1 else 0,
                documents=len(values),
                unit=unit,
            )
    word_lists = [doc.measurement.function_words for doc in documents]
    words = sorted({word for item in word_lists if item is not None for word in item.frequencies})
    for word in words:
        frequencies = [item.frequencies[word] for item in word_lists if item is not None and word in item.frequencies]
        if frequencies:
            result[f"function_words.{word}"] = NumericSummary(
                mean=mean(frequencies),
                median=median(frequencies),
                minimum=min(frequencies),
                maximum=max(frequencies),
                stdev=stdev(frequencies) if len(frequencies) > 1 else 0,
                documents=len(frequencies),
                unit="occurrences/1000 lexical words",
            )
    return result


def build_voice_profile(
    loaded: list[LoadedDocument],
    author: str,
    *,
    corrections: list[TraitCorrection] | None = None,
    annotations: list[VoiceTrait] | None = None,
) -> VoiceProfile:
    """Extract observations from the author's training split, retaining raw evidence."""
    from colophon.analysis.pipeline import analyze

    selected = sorted(
        (
            doc
            for doc in loaded
            if doc.entry.include and doc.entry.split == "train" and doc.entry.effective_author == author
        ),
        key=lambda doc: doc.entry.id,
    )
    if not selected:
        raise ValueError(f"No included training documents for author {author!r}")
    if len({doc.entry.language for doc in selected}) != 1:
        raise ValueError("A voice profile must have one language; build separate profiles")
    for attribute in ("id", "work_id"):
        values = [getattr(doc.entry, attribute) for doc in selected]
        if len(values) != len(set(values)):
            raise ValueError(f"Training pieces must have unique {attribute} values")
    if len({doc.normalized_hash for doc in selected}) != len(selected):
        raise ValueError("Duplicate normalized training documents cannot establish independent support")

    documents = []
    warnings = []
    for doc in selected:
        if not doc.author_document.text.strip():
            raise ValueError(f"No included author narration in {doc.entry.id}")
        measurement = analyze(doc.author_document, lang=doc.entry.language)
        documents.append(
            VoiceDocument(
                entry=doc.entry,
                source_path=doc.document.source_path,
                extraction_metadata=doc.document.metadata,
                source_hash=doc.source_hash,
                normalized_hash=doc.normalized_hash,
                measurement=measurement,
                author_word_count=measurement.word_count,
                excluded_word_count=doc.excluded_word_count,
            )
        )
        warnings.extend(f"{doc.entry.id}: {warning}" for warning in doc.warnings + measurement.warnings)
        if doc.entry.contributors:
            warnings.append(
                f"{doc.entry.id}: contributor context retained; editing and translation can affect measured voice."
            )

    traits = extract_traits(selected)
    trait_ids = {trait.id for trait in traits}
    for annotation in annotations or []:
        if annotation.id in trait_ids:
            raise ValueError(f"Duplicate trait ID: {annotation.id}")
        if not annotation.evidence:
            raise ValueError("An annotated observation requires exact supporting source evidence")
        for span in annotation.evidence + annotation.counterexamples:
            validate_evidence(span, selected)
        # Exact citations prove location, not the annotation's interpretation.
        traits.append(annotation.model_copy(update={"method": "annotation", "status": "tentative"}))
        trait_ids.add(annotation.id)
    for trait in traits:
        for span in trait.evidence + trait.counterexamples:
            validate_evidence(span, selected)

    applied = corrections or []
    if len({item.trait_id for item in applied}) != len(applied):
        raise ValueError("Provide only one correction per trait")
    by_id = {trait.id: trait for trait in traits}
    for correction in applied:
        if correction.trait_id not in by_id:
            raise ValueError(f"Unknown corrected trait: {correction.trait_id}")
        trait = by_id[correction.trait_id]
        updates: dict[str, object] = {"original_statement": trait.statement}
        if correction.action == "replace":
            if not correction.statement:
                raise ValueError("Replacing a trait requires a writer-supplied statement")
            updates.update(statement=correction.statement, status="writer_supplied")
        else:
            updates["status"] = "accepted" if correction.action == "accept" else "rejected"
        by_id[trait.id] = trait.model_copy(update=updates)
    traits = [by_id[trait.id] for trait in traits]

    # Greedy context coverage avoids selecting only the first sorted source group.
    # This is a diverse sample of eligible prose, not a style-optimality claim.
    pools = {doc.entry.id: author_evidence(doc, role="example") for doc in selected}
    contexts = {}
    for doc in selected:
        context = set()
        for name in ("genre", "topic"):
            value = getattr(doc.entry, name).strip().casefold()
            if value and value != "unspecified":
                context.add((name, value))
        year = re.search(r"\b(\d{4})\b", doc.entry.date or "")
        if year:
            context.add(("era", str(int(year.group(1)) // 10 * 10)))
        contexts[doc.entry.id] = context
    weights = {"genre": 4, "era": 2, "topic": 1}
    examples: list[EvidenceSpan] = []
    seen_text: set[str] = set()
    covered: set[tuple[str, str]] = set()
    used_this_round: set[str] = set()
    while len(examples) < 6:
        candidates = []
        for doc in selected:
            identifier = doc.entry.id
            if identifier in used_this_round:
                continue
            pool = pools[identifier]
            # Alternate the opening, middle and ending of the eligible passages.
            role = len(examples) % 3
            target = 0 if role == 0 else len(pool) // 2 if role == 1 else len(pool) - 1
            for index in sorted(range(len(pool)), key=lambda index: (abs(index - target), index)):
                span = pool[index]
                words = list(re.finditer(r"\S+", span.text))
                if len(words) > 120:
                    stop = words[119].end()
                    span = span.model_copy(update={"text": span.text[:stop], "end": span.start + stop})
                if span.text in seen_text:
                    continue
                gain = sum(weights[name] for name, value in contexts[identifier] - covered)
                position_available = len(pool) >= 3 or role == 0
                tie = hashlib.sha256(identifier.encode()).hexdigest()
                candidates.append((-gain, -int(position_available), tie, identifier, span))
                break
        if not candidates:
            if not used_this_round:
                break
            used_this_round.clear()
            continue
        _, _, _, identifier, span = min(candidates, key=lambda candidate: candidate[:4])
        validate_evidence(span, selected)
        examples.append(span)
        seen_text.add(span.text)
        covered.update(contexts[identifier])
        used_this_round.add(identifier)
    corpus_identity = [
        {
            "entry": doc.entry.model_dump(mode="json"),
            "source_hash": doc.source_hash,
            "normalized_hash": doc.normalized_hash,
            "author_text_hash": hashlib.sha256(doc.author_document.text.encode()).hexdigest(),
            "blocks": [block.model_dump(mode="json") for block in doc.document.blocks],
            "extraction_metadata": doc.document.metadata,
        }
        for doc in selected
    ]
    config = {
        "extractor_version": "1",
        "exemplar_selection": "diverse-context-2",
        "aggregation": "equal independent document",
        "rule_language": selected[0].entry.language,
        "measurements": [
            {
                "schema": d.measurement.schema_version,
                "version": d.measurement.analyzer_version,
                "model": d.measurement.spacy_model,
                "model_version": d.measurement.model_version,
                "spacy_version": d.measurement.spacy_version,
            }
            for d in documents
        ],
        "corrections": [item.model_dump(mode="json") for item in applied],
        "annotations": [item.model_dump(mode="json") for item in annotations or []],
    }
    corpus_hash, config_hash = _hash(corpus_identity), _hash(config)
    limitations = [
        "Observed patterns describe this training corpus, not a universal or topic-independent author identity.",
        "Literal markers do not establish humor, personality, beliefs, "
        "rhetorical intention or distinctiveness from peers.",
        "Numerical summaries weight independent pieces equally; per-document measurements retain variation.",
        "Quoted and excluded text is omitted from author measurements; "
        "quotation integration is a separate structural observation.",
        "Held-out voice quality and factual preservation require separate evaluation.",
    ]
    if len(selected) < 2:
        limitations.append("Only one independent work is available; recurring author habits cannot be established.")
    if selected[0].entry.language != "en":
        limitations.append("English argument/stance/transition marker rules were not applied to this language.")
    return VoiceProfile(
        profile_id=_hash({"corpus": corpus_hash, "config": config_hash}),
        author=author,
        language=selected[0].entry.language,
        corpus_hash=corpus_hash,
        config_hash=config_hash,
        documents=documents,
        numerical=_numeric_summaries(documents),
        traits=traits,
        exemplars=examples,
        corrections=applied,
        warnings=sorted(set(warnings)),
        limitations=limitations,
    )

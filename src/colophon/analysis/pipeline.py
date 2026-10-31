"""Analysis pipeline orchestrator."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from typing import Any

from colophon.analysis import load_spacy_model
from colophon.analysis.context import AnalysisContext, lexical_tokens
from colophon.models.document import Document
from colophon.models.features import StyleProfile

# Registry of available analyzers
ANALYZERS: dict[str, str] = {
    "readability": "colophon.analysis.readability",
    "sentences": "colophon.analysis.sentence",
    "vocabulary": "colophon.analysis.vocabulary",
    "function_words": "colophon.analysis.function_words",
    "pos": "colophon.analysis.pos",
    "ngrams": "colophon.analysis.ngrams",
    "punctuation": "colophon.analysis.punctuation",
    "contractions": "colophon.analysis.contractions",
    "sentence_openers": "colophon.analysis.openers",
    "paragraphs": "colophon.analysis.paragraphs",
    "dialogue": "colophon.analysis.dialogue",
    "syntax": "colophon.analysis.syntax",
}

# Analyzers that need spaCy
SPACY_ANALYZERS = {
    "sentences", "vocabulary", "function_words", "pos", "ngrams",
    "contractions", "sentence_openers", "paragraphs", "dialogue", "syntax", "punctuation",
}


def analyze(
    doc: Document,
    *,
    analyzers: list[str] | None = None,
    spacy_model: str = "en_core_web_sm",
    lang: str = "en",
    chunk_size: int = 50_000,
    workers: int | None = None,
) -> StyleProfile:
    """Measure an exact document using one shared parse.

    ``chunk_size`` and ``workers`` remain compatibility options. The former
    parallel implementation changed quote state, ngrams and distributions at
    chunk boundaries. Exact whole-document parsing replaces it until a bounded
    algorithm can preserve the same contract.
    """
    from colophon.embeddings.vectorize import VECTOR_SCHEMA
    from colophon.lang import get_profile

    if chunk_size < 1 or (workers is not None and workers < 1):
        raise ValueError("chunk_size and workers must be positive")
    if not doc.text.strip():
        raise ValueError("Cannot analyze empty text")
    if not any(character.isalnum() for character in doc.text):
        raise ValueError("Cannot analyze text without lexical words")
    lang_profile = get_profile(lang)
    active = list(ANALYZERS) if analyzers is None else list(dict.fromkeys(analyzers))
    unknown = set(active) - set(ANALYZERS)
    if unknown:
        raise ValueError(f"Unknown analyzers: {', '.join(sorted(unknown))}")
    availability = {name: "not_requested" for name in ANALYZERS}
    if lang_profile.textstat_code is None and "readability" in active:
        availability["readability"] = "not_applicable"
        active.remove("readability")
    if not lang_profile.contraction_suffixes and "contractions" in active:
        availability["contractions"] = "not_applicable"
        active.remove("contractions")

    needs_spacy = any(name in SPACY_ANALYZERS for name in active)
    context: Any = None
    model = spacy_model if spacy_model != "en_core_web_sm" else lang_profile.spacy_model
    model_metadata: dict[str, Any] = {}
    warnings = list(doc.warnings)
    if doc.metadata.get("paragraph_quality") == "unverified" and "paragraphs" in active:
        active.remove("paragraphs")
        availability["paragraphs"] = "unavailable"
        warnings.append("paragraphs: source paragraph boundaries are unverified")
    if needs_spacy:
        nlp = load_spacy_model(model)
        nlp.max_length = max(nlp.max_length, len(doc.text) + 1000)
        verified_paragraphs = doc.metadata.get("paragraph_quality") != "unverified"
        paragraph_ranges = None
        if doc.blocks and verified_paragraphs:
            paragraph_ranges = [
                (block.char_offset_start, block.char_offset_end)
                for block in doc.blocks if block.kind in {"paragraph", "quotation"}
            ]
        context = AnalysisContext(
            doc.text, nlp, paragraph_boundaries=verified_paragraphs,
            paragraph_ranges=paragraph_ranges,
        )
        model_metadata = nlp.meta
        annotations = {
            "pos": "POS", "sentence_openers": "POS", "syntax": "DEP",
        }
        if lang == "en":
            annotations["contractions"] = "POS"
        for name, annotation in annotations.items():
            if name in active and not context.doc.has_annotation(annotation):
                active.remove(name)
                availability[name] = "unavailable"
                warnings.append(f"{name}: model has no {annotation} annotation")
        if any(name in active for name in ("sentences", "paragraphs", "sentence_openers", "syntax")):
            if not context.doc.has_annotation("SENT_START"):
                raise ValueError("The selected spaCy model does not provide sentence boundaries")
        word_count = len(lexical_tokens(context.doc))
    else:
        word_count = doc.word_count
    if needs_spacy and doc.word_count > chunk_size:
        warnings.append("Exact whole-document parsing used; chunk_size/workers do not alter measurements")
    profile = StyleProfile(
        document_id=doc.id, document_title=doc.title, word_count=word_count,
        schema_version="2", analyzer_version="2", normalization_version="2",
        vector_schema=VECTOR_SCHEMA, language=lang,
        spacy_model=model if needs_spacy else None,
        spacy_version=version("spacy") if needs_spacy else None,
        model_version=model_metadata.get("version") if needs_spacy else None,
        source_hash=hashlib.sha256(doc.text.encode("utf-8")).hexdigest(),
        availability=availability, warnings=warnings,
    )
    for name in active:
        result = _run_analyzer(name, doc.text, context, lang_profile)
        if result is not None:
            availability[name] = "measured"
            profile = profile.model_copy(update={name: result})
    return profile.model_copy(update={"availability": availability})


def _run_analyzer(name: str, text: str, nlp: Any, lang_profile: Any = None) -> Any:
    """Run a single analyzer by name."""
    if name == "readability":
        from colophon.analysis.readability import compute_readability
        textstat_code = lang_profile.textstat_code if lang_profile else "en"
        return compute_readability(text, lang=textstat_code)
    elif name == "sentences":
        from colophon.analysis.sentence import compute_sentence_stats
        return compute_sentence_stats(text, nlp)
    elif name == "vocabulary":
        from colophon.analysis.vocabulary import compute_vocabulary
        return compute_vocabulary(text, nlp)
    elif name == "function_words":
        from colophon.analysis.function_words import compute_function_words
        word_list = lang_profile.function_words if lang_profile else None
        return compute_function_words(text, nlp, function_word_list=word_list)
    elif name == "pos":
        from colophon.analysis.pos import compute_pos
        return compute_pos(text, nlp)
    elif name == "ngrams":
        from colophon.analysis.ngrams import compute_ngrams
        return compute_ngrams(text, nlp)
    elif name == "punctuation":
        from colophon.analysis.punctuation import compute_punctuation
        word_count = len(lexical_tokens(nlp(text)))
        return compute_punctuation(text, word_count)
    elif name == "contractions":
        from colophon.analysis.contractions import compute_contractions
        suffixes = lang_profile.contraction_suffixes if lang_profile else None
        return compute_contractions(text, nlp, contraction_suffixes=suffixes)
    elif name == "sentence_openers":
        from colophon.analysis.openers import compute_openers
        return compute_openers(text, nlp)
    elif name == "paragraphs":
        from colophon.analysis.paragraphs import compute_paragraphs
        return compute_paragraphs(text, nlp)
    elif name == "dialogue":
        from colophon.analysis.dialogue import compute_dialogue
        quote_marks = lang_profile.quote_marks if lang_profile else None
        return compute_dialogue(text, nlp, quote_marks=quote_marks)
    elif name == "syntax":
        from colophon.analysis.syntax import compute_syntax
        return compute_syntax(text, nlp)
    return None

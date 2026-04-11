"""Analysis pipeline orchestrator."""

from __future__ import annotations

from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn

from colophon.analysis import load_spacy_model
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
    "contractions", "sentence_openers", "paragraphs", "dialogue", "syntax",
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
    """Run the analysis pipeline on a document.

    For documents over chunk_size words, automatically uses multiprocessing
    to parallelize spaCy-dependent analyzers across text chunks.
    """
    from colophon.lang import get_profile
    lang_profile = get_profile(lang)

    active = analyzers or list(ANALYZERS.keys())

    # Skip readability if the language doesn't support it
    if lang_profile.textstat_code is None and "readability" in active:
        active = [a for a in active if a != "readability"]

    needs_spacy = any(a in SPACY_ANALYZERS for a in active)

    # Auto-route to chunked processing for large documents
    if needs_spacy and doc.word_count > chunk_size:
        from colophon.analysis.chunked import analyze_chunked
        return analyze_chunked(
            doc, analyzers=active, spacy_model=spacy_model,
            lang=lang, chunk_size=chunk_size, workers=workers,
        )

    # Standard single-pass for smaller documents
    nlp: Any = None
    if needs_spacy:
        model = spacy_model if spacy_model != "en_core_web_sm" else lang_profile.spacy_model
        nlp = load_spacy_model(model)
        nlp.max_length = len(doc.text) + 1000

    profile = StyleProfile(
        document_id=doc.id,
        document_title=doc.title,
        word_count=doc.word_count,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        for name in active:
            if name not in ANALYZERS:
                continue
            progress.add_task(f"Running {name}...", total=None)
            result = _run_analyzer(name, doc.text, nlp, lang_profile)
            if result is not None:
                profile = profile.model_copy(update={name: result})

    return profile


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
        word_count = len(text.split())
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

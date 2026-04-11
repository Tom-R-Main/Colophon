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
}

# Analyzers that need spaCy
SPACY_ANALYZERS = {"sentences", "vocabulary", "function_words", "pos", "ngrams"}


def analyze(
    doc: Document,
    *,
    analyzers: list[str] | None = None,
    spacy_model: str = "en_core_web_sm",
) -> StyleProfile:
    """Run the analysis pipeline on a document."""
    active = analyzers or list(ANALYZERS.keys())
    needs_spacy = any(a in SPACY_ANALYZERS for a in active)

    nlp: Any = None
    if needs_spacy:
        nlp = load_spacy_model(spacy_model)
        # Optimize: disable components we don't need
        # We need tagger (POS), senter (sentences), and attribute_ruler
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
            result = _run_analyzer(name, doc.text, nlp)
            if result is not None:
                profile = profile.model_copy(update={name: result})

    return profile


def _run_analyzer(name: str, text: str, nlp: Any) -> Any:
    """Run a single analyzer by name."""
    if name == "readability":
        from colophon.analysis.readability import compute_readability
        return compute_readability(text)
    elif name == "sentences":
        from colophon.analysis.sentence import compute_sentence_stats
        return compute_sentence_stats(text, nlp)
    elif name == "vocabulary":
        from colophon.analysis.vocabulary import compute_vocabulary
        return compute_vocabulary(text, nlp)
    elif name == "function_words":
        from colophon.analysis.function_words import compute_function_words
        return compute_function_words(text, nlp)
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
    return None

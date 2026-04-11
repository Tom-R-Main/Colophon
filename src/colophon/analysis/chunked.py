"""Chunked multiprocessing analysis for large documents."""

from __future__ import annotations

import os
from collections import Counter
from multiprocessing import Pool
from typing import Any

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from colophon.analysis.merge import (
    merge_contraction_features,
    merge_dialogue_features,
    merge_function_word_features,
    merge_ngram_features,
    merge_opener_features,
    merge_pos_features,
    merge_sentence_features,
    merge_syntax_features,
    merge_vocabulary_features,
)
from colophon.analysis.pipeline import SPACY_ANALYZERS, _run_analyzer
from colophon.models.document import Document
from colophon.models.features import StyleProfile

CHUNK_TARGET = 50_000  # words per chunk


def chunk_text(text: str, target_words: int = CHUNK_TARGET) -> list[str]:
    """Split text at paragraph boundaries into roughly equal chunks."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        words = len(para.split())
        if words == 0:
            continue
        current.append(para)
        current_words += words
        if current_words >= target_words:
            chunks.append("\n\n".join(current))
            current = []
            current_words = 0

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _analyze_chunk_worker(args: tuple) -> dict[str, Any]:
    """Worker function for multiprocessing — runs in a separate process.

    Each worker loads its own spaCy model (can't pickle nlp objects).
    Returns raw partial data (counters, lists) for merging.
    """
    chunk_text_str, spacy_model, lang_code, analyzer_names = args

    from colophon.analysis import load_spacy_model
    nlp = load_spacy_model(spacy_model)
    nlp.max_length = len(chunk_text_str) + 1000
    doc = nlp(chunk_text_str)

    results: dict[str, Any] = {}

    if "sentences" in analyzer_names:
        lengths = [len([t for t in sent if not t.is_space]) for sent in doc.sents]
        results["sentences"] = {"lengths": lengths}

    if "vocabulary" in analyzer_names:
        tokens = [t.lower_ for t in doc if not t.is_punct and not t.is_space]
        results["vocabulary"] = {
            "freq_dist": dict(Counter(tokens)),
            "total_tokens": len(tokens),
        }

    if "pos" in analyzer_names:
        tokens = [t for t in doc if not t.is_space]
        results["pos"] = {"pos_counts": dict(Counter(t.pos_ for t in tokens))}

    if "function_words" in analyzer_names:
        from colophon.lang import get_profile
        lang_profile = get_profile(lang_code)
        word_list = lang_profile.function_words
        tokens = [t.lower_ for t in doc if not t.is_space]
        token_counts = Counter(tokens)
        fw_counts = {w: token_counts.get(w, 0) for w in word_list if token_counts.get(w, 0) > 0}
        results["function_words"] = {"fw_counts": fw_counts, "total_tokens": len(tokens)}

    if "ngrams" in analyzer_names:
        tokens = [t.lower_ for t in doc if not t.is_punct and not t.is_space]
        word_bi = Counter(" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)) if len(tokens) > 1 else Counter()
        word_tri = Counter(" ".join(tokens[i:i + 3]) for i in range(len(tokens) - 2)) if len(tokens) > 2 else Counter()
        clean = " ".join(chunk_text_str.lower().split())
        char_tri = Counter(clean[i:i + 3] for i in range(len(clean) - 2)) if len(clean) > 2 else Counter()
        results["ngrams"] = {
            "word_bigrams": dict(word_bi.most_common(200)),
            "word_trigrams": dict(word_tri.most_common(200)),
            "char_trigrams": dict(char_tri.most_common(200)),
        }

    if "contractions" in analyzer_names:
        from colophon.lang import get_profile
        lang_profile = get_profile(lang_code)
        suffixes = set(lang_profile.contraction_suffixes) if lang_profile.contraction_suffixes else set()
        contraction_counts: Counter[str] = Counter()
        total_words = len([t for t in doc if not t.is_space and not t.is_punct])
        for token in doc:
            if token.text.lower() in suffixes and token.i > 0:
                full = doc[token.i - 1].text.lower() + token.text.lower()
                contraction_counts[full] += 1
        results["contractions"] = {"contraction_counts": dict(contraction_counts), "total_words": total_words}

    if "sentence_openers" in analyzer_names:
        opener_words: Counter[str] = Counter()
        opener_pos: Counter[str] = Counter()
        conj_starts = 0
        total_sents = 0
        for sent in doc.sents:
            tokens = [t for t in sent if not t.is_space]
            if not tokens:
                continue
            total_sents += 1
            opener_words[tokens[0].lower_] += 1
            opener_pos[tokens[0].pos_] += 1
            if tokens[0].pos_ == "CCONJ":
                conj_starts += 1
        results["sentence_openers"] = {
            "opener_words": dict(opener_words),
            "opener_pos": dict(opener_pos),
            "conjunction_starts": conj_starts,
            "total_sents": total_sents,
        }

    if "dialogue" in analyzer_names:
        from colophon.lang import get_profile
        lang_profile = get_profile(lang_code)
        qmarks = set(lang_profile.quote_marks) if lang_profile.quote_marks else {'"'}
        in_quote = False
        quoted_words = 0
        narration_words = 0
        attribution_verbs: Counter[str] = Counter()
        for i, token in enumerate(doc):
            if token.text in qmarks:
                in_quote = not in_quote
                window = doc[max(0, i - 5):min(len(doc), i + 6)]
                for t in window:
                    if t.pos_ == "VERB" and t.dep_ in ("ROOT", "ccomp", "parataxis"):
                        attribution_verbs[t.lemma_] += 1
                continue
            if token.is_space or token.is_punct:
                continue
            if in_quote:
                quoted_words += 1
            else:
                narration_words += 1
        results["dialogue"] = {
            "quoted_words": quoted_words,
            "narration_words": narration_words,
            "attribution_verbs": dict(attribution_verbs),
        }

    if "syntax" in analyzer_names:
        depths: list[int] = []
        tense_counts: Counter[str] = Counter()
        type_counts: Counter[str] = Counter()

        for sent in doc.sents:
            roots = [t for t in sent if t.dep_ == "ROOT"]
            if roots:
                depths.append(_tree_depth(roots[0]))
            text_s = sent.text.strip()
            if text_s.endswith("?"):
                type_counts["interrogative"] += 1
            elif text_s.endswith("!"):
                type_counts["exclamatory"] += 1
            else:
                type_counts["declarative"] += 1

        for token in doc:
            if token.pos_ in ("VERB", "AUX"):
                morph = token.morph.to_dict()
                tense = morph.get("Tense", "None")
                tense_counts[tense] += 1

        results["syntax"] = {
            "depths": depths,
            "tense_counts": dict(tense_counts),
            "type_counts": dict(type_counts),
        }

    return results


def _tree_depth(token: Any, depth: int = 0) -> int:
    children_depths = [_tree_depth(child, depth + 1) for child in token.children]
    return max(children_depths) if children_depths else depth


def analyze_chunked(
    doc: Document,
    *,
    analyzers: list[str],
    spacy_model: str,
    lang: str,
    chunk_size: int = CHUNK_TARGET,
    workers: int | None = None,
) -> StyleProfile:
    """Analyze a large document by chunking and processing in parallel."""
    from colophon.lang import get_profile
    lang_profile = get_profile(lang)

    # Separate spaCy vs non-spaCy analyzers
    spacy_analyzers = [a for a in analyzers if a in SPACY_ANALYZERS]
    non_spacy_analyzers = [a for a in analyzers if a not in SPACY_ANALYZERS]

    # Resolve spaCy model
    model = spacy_model if spacy_model != "en_core_web_sm" else lang_profile.spacy_model

    # Chunk the text
    chunks = chunk_text(doc.text, chunk_size)
    num_workers = workers or min(os.cpu_count() or 4, len(chunks), 8)

    profile = StyleProfile(
        document_id=doc.id,
        document_title=doc.title,
        word_count=doc.word_count,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        transient=True,
    ) as progress:
        # Run non-spaCy analyzers on full text (fast)
        if non_spacy_analyzers:
            task_id = progress.add_task("Non-spaCy analyzers...", total=len(non_spacy_analyzers))
            for name in non_spacy_analyzers:
                result = _run_analyzer(name, doc.text, None, lang_profile)
                if result is not None:
                    profile = profile.model_copy(update={name: result})
                progress.advance(task_id)

        # Run spaCy analyzers chunked in parallel
        if spacy_analyzers:
            task_id = progress.add_task(
                f"spaCy analyzers ({len(chunks)} chunks, {num_workers} workers)...",
                total=len(chunks),
            )

            args = [(chunk, model, lang, spacy_analyzers) for chunk in chunks]

            with Pool(num_workers) as pool:
                partial_results = []
                for result in pool.imap_unordered(_analyze_chunk_worker, args):
                    partial_results.append(result)
                    progress.advance(task_id)

            # Merge partial results
            progress.add_task("Merging results...", total=None)
            merged = _merge_all(partial_results, spacy_analyzers)
            for name, feature in merged.items():
                profile = profile.model_copy(update={name: feature})

        # Paragraph analysis runs separately (needs full text for paragraph boundaries)
        if "paragraphs" in analyzers:
            progress.add_task("Paragraph analysis...", total=None)
            from colophon.analysis import load_spacy_model
            nlp = load_spacy_model(model)
            nlp.max_length = len(doc.text) + 1000
            result = _run_analyzer("paragraphs", doc.text, nlp, lang_profile)
            if result is not None:
                profile = profile.model_copy(update={"paragraphs": result})

    return profile


def _merge_all(partials: list[dict], analyzer_names: list[str]) -> dict[str, Any]:
    """Merge all partial results from chunked workers."""
    merged: dict[str, Any] = {}

    if "sentences" in analyzer_names:
        sentence_partials = [p["sentences"] for p in partials if "sentences" in p]
        if sentence_partials:
            merged["sentences"] = merge_sentence_features(sentence_partials)

    if "vocabulary" in analyzer_names:
        vocab_partials = [p["vocabulary"] for p in partials if "vocabulary" in p]
        if vocab_partials:
            merged["vocabulary"] = merge_vocabulary_features(vocab_partials)

    if "pos" in analyzer_names:
        pos_partials = [p["pos"] for p in partials if "pos" in p]
        if pos_partials:
            merged["pos"] = merge_pos_features(pos_partials)

    if "function_words" in analyzer_names:
        fw_partials = [p["function_words"] for p in partials if "function_words" in p]
        if fw_partials:
            merged["function_words"] = merge_function_word_features(fw_partials)

    if "ngrams" in analyzer_names:
        ng_partials = [p["ngrams"] for p in partials if "ngrams" in p]
        if ng_partials:
            merged["ngrams"] = merge_ngram_features(ng_partials)

    if "contractions" in analyzer_names:
        ct_partials = [p["contractions"] for p in partials if "contractions" in p]
        if ct_partials:
            merged["contractions"] = merge_contraction_features(ct_partials)

    if "sentence_openers" in analyzer_names:
        op_partials = [p["sentence_openers"] for p in partials if "sentence_openers" in p]
        if op_partials:
            merged["sentence_openers"] = merge_opener_features(op_partials)

    if "dialogue" in analyzer_names:
        dl_partials = [p["dialogue"] for p in partials if "dialogue" in p]
        if dl_partials:
            merged["dialogue"] = merge_dialogue_features(dl_partials)

    if "syntax" in analyzer_names:
        sx_partials = [p["syntax"] for p in partials if "syntax" in p]
        if sx_partials:
            merged["syntax"] = merge_syntax_features(sx_partials)

    return merged

"""Correctness checks at numerical library boundaries using real spaCy docs."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
import spacy

from colophon.analysis import pipeline
from colophon.analysis.chunked import analyze_chunked
from colophon.analysis.contractions import compute_contractions
from colophon.analysis.function_words import compute_function_words
from colophon.analysis.paragraphs import compute_paragraphs
from colophon.analysis.sentence import compute_sentence_stats
from colophon.comparison.burrows_delta import compare
from colophon.embeddings.vectorize import VECTOR_SCHEMA, style_profile_to_vector
from colophon.lang import get_profile
from colophon.models.document import Corpus, Document, DocumentBlock
from colophon.models.features import FunctionWordFeatures, PunctuationFeatures, StyleProfile


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def real_pipeline(monkeypatch, nlp):
    monkeypatch.setattr(pipeline, "load_spacy_model", lambda name: nlp)
    return pipeline.analyze


def canonical_profile(values=None, **kwargs):
    frequencies = dict.fromkeys(get_profile("en").function_words, 0.0)
    frequencies.update(values or {})
    return StyleProfile(
        document_id="fixture", document_title="fixture", schema_version="2", analyzer_version="2",
        language="en", vector_schema=VECTOR_SCHEMA,
        function_words=FunctionWordFeatures(frequencies=frequencies),
        availability={"function_words": "measured"}, **kwargs,
    )


def test_function_word_identity_is_a_fixed_coordinate():
    first = canonical_profile({"and": 40, "the": 60})
    second = canonical_profile({"but": 40, "of": 60})
    a, b = style_profile_to_vector(first), style_profile_to_vector(second)
    assert len(a) == len(b) == 128
    assert a != b
    assert sum(x * y for x, y in zip(a, b)) < 0.01
    reordered = first.model_copy(update={
        "function_words": FunctionWordFeatures(
            frequencies=dict(reversed(list(first.function_words.frequencies.items()))),
        ),
    })
    assert style_profile_to_vector(reordered) == a


def test_legacy_and_sparse_profiles_are_rejected():
    with pytest.raises(ValueError, match="re-analyze"):
        style_profile_to_vector(StyleProfile(document_id="old", document_title="old"))
    incomplete = canonical_profile().model_copy(update={
        "function_words": FunctionWordFeatures(frequencies={"the": 5}),
    })
    with pytest.raises(ValueError, match="canonical"):
        style_profile_to_vector(incomplete)


def test_measured_zero_is_distinct_from_unavailable():
    measured = canonical_profile().model_copy(update={
        "punctuation": PunctuationFeatures(comma=0),
        "availability": {"function_words": "measured", "punctuation": "measured"},
    })
    unavailable = measured.model_copy(update={
        "function_words": None, "availability": {"function_words": "unavailable", "punctuation": "measured"},
    })
    assert style_profile_to_vector(measured) != style_profile_to_vector(unavailable)
    invalid = measured.model_copy(update={"availability": {"punctuation": "measured"}})
    with pytest.raises(ValueError, match="availability"):
        style_profile_to_vector(invalid)
    nonfinite = canonical_profile({"the": math.nan})
    with pytest.raises(ValueError, match="finite"):
        style_profile_to_vector(nonfinite)


def test_punctuation_does_not_inflate_word_units(nlp):
    plain = compute_function_words("The cat", nlp)
    punctuated = compute_function_words("The , cat .", nlp)
    assert plain.frequencies["the"] == punctuated.frequencies["the"] == 500
    assert len(plain.frequencies) == len(get_profile("en").function_words)
    assert plain.frequencies["but"] == 0
    assert compute_function_words("The cat", nlp, function_word_list=[]).frequencies == {}
    assert compute_sentence_stats("The cat.", nlp).mean_length == 2


def test_short_paragraphs_are_kept(nlp, real_pipeline):
    text = "No.\n\nNever.\n\nThis paragraph is long enough."
    direct = compute_paragraphs(text, nlp)
    assert direct.count == 3
    profile = real_pipeline(Document.from_text(text, "short.txt"), analyzers=["paragraphs", "sentences"])
    assert profile.paragraphs.count == 3
    assert profile.paragraphs.one_sentence_ratio == 1
    assert profile.sentences.count == 3
    assert profile.sentences.min_length == 1


def test_contractions_are_not_possessives(nlp):
    assert compute_contractions("John's dog ran.", nlp).total == 0
    assert compute_contractions("He's leaving. John's going home.", nlp).total == 2
    assert compute_contractions("He's leaving.", nlp, contraction_suffixes=[]).total == 0


def test_prefix_elision_at_first_token():
    french = spacy.blank("fr")
    result = compute_contractions("l'homme arrive", french, contraction_suffixes=["l'"])
    assert result.top_contractions == {"l'homme": 1}


def test_shared_parse_and_metadata(monkeypatch, nlp):
    class CountingModel:
        meta = nlp.meta
        max_length = nlp.max_length

        def __init__(self):
            self.calls = 0

        def make_doc(self, text):
            return nlp.make_doc(text)

        def __call__(self, value):
            self.calls += 1
            return nlp(value)

    model = CountingModel()
    monkeypatch.setattr(pipeline, "load_spacy_model", lambda name: model)
    text = "The cat sleeps.\n\nBut I do not."
    profile = pipeline.analyze(Document.from_text(text, "parsed.txt"))
    assert model.calls == 1
    assert profile.schema_version == profile.analyzer_version == "2"
    assert profile.language == "en" and profile.vector_schema == VECTOR_SCHEMA
    assert profile.spacy_version and profile.model_version == nlp.meta["version"]
    assert profile.source_hash == hashlib.sha256(text.encode()).hexdigest()
    assert set(profile.availability.values()) == {"measured"}
    assert len(style_profile_to_vector(profile)) == 128


def test_chunk_size_and_workers_cannot_change_features(real_pipeline):
    text = '"One small sentence.\n\nAnd a longer sentence inside this quote." The narrator speaks.\n\nNo.'
    document = Document.from_text(text, "quote.txt")
    expected = real_pipeline(document, chunk_size=1000, workers=1)
    for chunk_size, workers in [(1, 1), (2, 2), (100, 8)]:
        actual = analyze_chunked(
            document, analyzers=list(pipeline.ANALYZERS), spacy_model="en_core_web_sm",
            lang="en", chunk_size=chunk_size, workers=workers,
        )
        for feature in pipeline.ANALYZERS:
            assert getattr(actual, feature) == getattr(expected, feature), feature
    assert expected.word_count == 14
    assert expected.dialogue.quoted_word_ratio == round(10 / 14, 3)
    assert expected.ngrams.word_bigrams["sentence and"] == 1


def test_missing_annotations_and_source_quality_are_visible(monkeypatch):
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    monkeypatch.setattr(pipeline, "load_spacy_model", lambda name: nlp)
    document = Document.from_text(
        "A paragraph.\n\nAnother paragraph.", "layout.pdf",
        metadata={"paragraph_quality": "unverified"}, warnings=["layout not qualified"],
    )
    result = pipeline.analyze(document, analyzers=["paragraphs", "syntax", "pos", "sentences"])
    assert result.paragraphs is result.syntax is result.pos is None
    assert result.availability["paragraphs"] == result.availability["syntax"] == "unavailable"
    assert result.availability["sentences"] == "measured"
    assert "layout not qualified" in result.warnings


def test_empty_selection_and_invalid_requests(monkeypatch):
    monkeypatch.setattr(pipeline, "load_spacy_model", lambda name: pytest.fail("no parser should be loaded"))
    document = Document.from_text("A text.", "text.txt")
    result = pipeline.analyze(document, analyzers=[])
    assert set(result.availability.values()) == {"not_requested"}
    with pytest.raises(ValueError, match="numeric features"):
        style_profile_to_vector(result)
    with pytest.raises(ValueError, match="lexical words"):
        pipeline.analyze(Document.from_text("...", "punctuation.txt"))
    with pytest.raises(ValueError, match="Unknown analyzers"):
        pipeline.analyze(document, analyzers=["typo"])
    with pytest.raises(ValueError, match="empty"):
        pipeline.analyze(Document.from_text("  ", "empty.txt"))
    with pytest.raises(ValueError, match="positive"):
        pipeline.analyze(document, workers=0)


def references():
    return Corpus(name="reference", documents=[
        Document.from_text("the and the of", "a1", author="A"),
        Document.from_text("and the of the", "a2", author="A"),
        Document.from_text("of but of and", "b1", author="B"),
        Document.from_text("but of but and", "b2", author="B"),
    ])


def test_delta_reference_fit_matches_independent_calculation(monkeypatch):
    corpus = references()
    unknown = Document.from_text("the the and and", "query")
    result = compare(corpus, unknown, n_features=100)
    # Independently specified sorted reference vocabulary and document-level scales.
    vocabulary = ["the", "of", "and", "but"]
    raw = np.array([
        [0.5, 0.25, 0.25, 0], [0.5, 0.25, 0.25, 0],
        [0, 0.5, 0.25, 0.25], [0, 0.25, 0.25, 0.5],
    ])
    scales = raw.std(axis=0)
    scales[scales == 0] = 1
    query = np.array([0.5, 0, 0.5, 0])
    independent = {
        "A": np.abs((query - raw[:2].mean(axis=0)) / scales).mean(),
        "B": np.abs((query - raw[2:].mean(axis=0)) / scales).mean(),
    }
    assert result.n_features == 4
    reversed_corpus = corpus.model_copy(update={"documents": list(reversed(corpus.documents))})
    assert compare(reversed_corpus, unknown).ranked_authors == result.ranked_authors
    assert dict(result.ranked_authors) == {name: round(float(value), 4) for name, value in independent.items()}
    seen = []
    import colophon.comparison.burrows_delta as delta
    original = delta._frequency_vector

    def capture(text, words):
        seen.append(tuple(words))
        return original(text, words)

    monkeypatch.setattr(delta, "_frequency_vector", capture)
    compare(corpus, Document.from_text("zebra zebra zebra zebra zebra", "novel-topic"))
    assert all(set(words) == set(vocabulary) for words in seen)


def test_delta_rejects_invalid_or_leaked_inputs():
    corpus = references()
    query = Document.from_text("A new query", "query")
    for invalid in [Corpus(name="empty"), Corpus(name="one", documents=corpus.documents[:2])]:
        with pytest.raises(ValueError):
            compare(invalid, query)
    with pytest.raises(ValueError, match="reference corpus"):
        compare(corpus, corpus.documents[0])
    with pytest.raises(ValueError, match="no lexical"):
        compare(corpus, Document.from_text("...", "empty"))
    with pytest.raises(ValueError, match="positive"):
        compare(corpus, query, n_features=0)


def test_reviewed_block_boundaries_override_single_newline_projection(real_pipeline):
    first, second = "First body paragraph", "Second body paragraph"
    text = first + "\n" + second
    blocks = [
        DocumentBlock(
            id="first", kind="paragraph", text=first,
            char_offset_start=0, char_offset_end=len(first),
        ),
        DocumentBlock(
            id="second", kind="paragraph", text=second,
            char_offset_start=len(first) + 1, char_offset_end=len(text),
        ),
    ]
    supplied = Document.from_text(text, "reviewed.colophon.json", blocks=blocks)
    imported = Document.model_validate_json(supplied.model_dump_json())
    expected = real_pipeline(imported, analyzers=["paragraphs", "sentences"], chunk_size=100)
    assert expected.paragraphs.count == 2
    assert expected.paragraphs.mean_length == expected.paragraphs.median_length == 3
    assert expected.paragraphs.one_sentence_ratio == 1
    assert expected.sentences.count == 2
    assert expected.sentences.mean_length == 3
    for chunk_size, workers in [(1, 1), (2, 2)]:
        result = analyze_chunked(
            imported, analyzers=["paragraphs", "sentences"], spacy_model="en_core_web_sm",
            lang="en", chunk_size=chunk_size, workers=workers,
        )
        assert result.paragraphs == expected.paragraphs
        assert result.sentences == expected.sentences
    assert imported.blocks == blocks and imported.text == text
    unverified = imported.model_copy(update={"metadata": {"paragraph_quality": "unverified"}})
    refused = real_pipeline(unverified, analyzers=["paragraphs"])
    assert refused.paragraphs is None and refused.availability["paragraphs"] == "unavailable"


def test_paragraph_projection_excludes_nonbody_blocks(real_pipeline):
    heading, body, furniture = "A heading", "The body paragraph", "Page footer"
    text = "\n".join([heading, body, furniture])
    blocks = []
    start = 0
    for index, (kind, value) in enumerate([("heading", heading), ("paragraph", body), ("furniture", furniture)]):
        blocks.append(DocumentBlock(
            id=str(index), kind=kind, text=value, char_offset_start=start, char_offset_end=start + len(value),
        ))
        start += len(value) + 1
    document = Document.from_text(text, "structured.txt", blocks=blocks)
    result = real_pipeline(document, analyzers=["paragraphs"])
    assert result.paragraphs.count == 1 and result.paragraphs.mean_length == 3
    nonbody_only = document.model_copy(update={"blocks": [blocks[0], blocks[2]]})
    assert real_pipeline(nonbody_only, analyzers=["paragraphs"]).paragraphs.count == 0


def test_context_explicit_ranges_preserve_offsets_and_validate_alignment(nlp):
    from colophon.analysis.context import AnalysisContext

    text = "First body paragraph\nSecond body paragraph"
    ranges = [(0, 20), (21, len(text))]
    context = AnalysisContext(text, nlp, paragraph_ranges=ranges)
    assert [(span.start_char, span.end_char) for span in context.paragraph_spans()] == ranges
    assert context.text == text and context.paragraph_ranges == ranges
    with pytest.raises(ValueError, match="ordered"):
        AnalysisContext(text, nlp, paragraph_ranges=[(21, len(text)), (0, 20)])
    with pytest.raises(ValueError, match="token boundaries"):
        AnalysisContext(text, nlp, paragraph_ranges=[(1, 20)])

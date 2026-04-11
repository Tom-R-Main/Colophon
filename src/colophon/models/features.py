"""Feature models for stylometric analysis results."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ReadabilityFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog: float
    coleman_liau: float
    ari: float
    smog: float | None = None  # Requires 30+ sentences


class SentenceFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    mean_length: float
    median_length: float
    stdev_length: float
    skewness: float
    min_length: int
    max_length: int
    length_distribution: list[int] = Field(default_factory=list, description="Histogram of sentence lengths.")


class VocabularyFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_tokens: int
    unique_types: int
    ttr: float = Field(description="Type-token ratio (length-biased).")
    hapax_legomena: int = Field(description="Words appearing exactly once.")
    hapax_ratio: float
    yules_k: float = Field(description="Yule's K — vocabulary richness, length-independent.")
    honores_r: float | None = Field(None, description="Honore's R — rewards hapax legomena. None if V1 == V.")


class FunctionWordFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    frequencies: dict[str, float] = Field(description="Function word -> per-1000-words frequency.")
    top_n: int = 30


class POSFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    tag_distribution: dict[str, float] = Field(description="POS tag -> proportion.")
    adjective_noun_ratio: float
    adverb_density: float = Field(description="Adverbs per 100 words.")
    verb_density: float = Field(description="Verbs per 100 words.")


class NGramFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    word_bigrams: dict[str, int] = Field(default_factory=dict, description="Top word bigrams.")
    word_trigrams: dict[str, int] = Field(default_factory=dict, description="Top word trigrams.")
    char_trigrams: dict[str, int] = Field(default_factory=dict, description="Top character trigrams.")


class PunctuationFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    comma: float = Field(description="Per 1000 words.")
    period: float = 0.0
    semicolon: float = 0.0
    colon: float = 0.0
    dash: float = Field(0.0, description="Em-dash and double-dash combined.")
    exclamation: float = 0.0
    question: float = 0.0
    ellipsis: float = 0.0
    parenthesis: float = 0.0
    quotation: float = 0.0


class ContractionFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: int
    rate_per_1000: float
    top_contractions: dict[str, int] = Field(default_factory=dict, description="Contraction -> count.")


class SentenceOpenerFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    top_words: dict[str, int] = Field(default_factory=dict, description="Opener word -> count.")
    pos_distribution: dict[str, float] = Field(default_factory=dict, description="Opener POS -> proportion.")
    conjunction_start_rate: float = Field(description="% of sentences starting with CCONJ (But, And, So).")


class ParagraphFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    mean_length: float = Field(description="Words per paragraph.")
    median_length: float
    one_sentence_ratio: float = Field(description="% of paragraphs that are a single sentence.")
    top_openers: dict[str, int] = Field(default_factory=dict, description="Paragraph-initial word -> count.")


class DialogueFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    quoted_word_ratio: float = Field(description="% of words that are inside quotes.")
    narration_word_ratio: float
    top_attribution_verbs: dict[str, int] = Field(default_factory=dict, description="Attribution verb -> count.")


class SyntaxFeatures(BaseModel):
    model_config = ConfigDict(frozen=True)

    mean_tree_depth: float = Field(description="Mean dependency tree depth per sentence.")
    median_tree_depth: float
    stdev_tree_depth: float
    tense_distribution: dict[str, float] = Field(default_factory=dict, description="Tense -> proportion of verbs.")
    sentence_type_mix: dict[str, float] = Field(
        default_factory=dict, description="declarative/interrogative/exclamatory -> proportion."
    )


class StyleProfile(BaseModel):
    """Complete stylometric profile for a document or segment."""

    document_id: str
    document_title: str
    segment_id: int | None = None
    computed_at: datetime = Field(default_factory=datetime.now)
    word_count: int = 0

    readability: ReadabilityFeatures | None = None
    sentences: SentenceFeatures | None = None
    vocabulary: VocabularyFeatures | None = None
    function_words: FunctionWordFeatures | None = None
    pos: POSFeatures | None = None
    ngrams: NGramFeatures | None = None
    punctuation: PunctuationFeatures | None = None
    contractions: ContractionFeatures | None = None
    sentence_openers: SentenceOpenerFeatures | None = None
    paragraphs: ParagraphFeatures | None = None
    dialogue: DialogueFeatures | None = None
    syntax: SyntaxFeatures | None = None


class ComparisonResult(BaseModel):
    """Result of authorship comparison via Burrows' Delta."""

    unknown_document_id: str
    unknown_document_title: str
    ranked_authors: list[tuple[str, float]] = Field(description="(author, delta_score) sorted by ascending delta.")
    n_features: int
    method: str = "burrows_delta"

"""Language profiles for multilingual stylometric analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LanguageProfile:
    """Configuration bundle for language-specific analysis."""

    code: str
    name: str
    spacy_model: str
    textstat_code: str | None  # None = skip readability
    quote_marks: list[str] = field(default_factory=lambda: ['"'])
    contraction_suffixes: list[str] = field(default_factory=list)
    function_words: list[str] = field(default_factory=list)


# Tier 1: Full support (spaCy + textstat + internationalizer)
PROFILES: dict[str, LanguageProfile] = {
    "en": LanguageProfile(
        code="en",
        name="English",
        spacy_model="en_core_web_sm",
        textstat_code="en",
        quote_marks=['"', "\u201c", "\u201d"],
        contraction_suffixes=["n't", "'s", "'m", "'re", "'ve", "'ll", "'d"],
        function_words=[
            "the", "of", "and", "to", "a", "in", "that", "is", "was", "it",
            "for", "as", "with", "his", "he", "on", "be", "at", "by", "i",
            "this", "had", "not", "are", "but", "from", "or", "have", "an", "they",
            "which", "one", "you", "were", "her", "all", "she", "there", "would", "their",
            "we", "him", "been", "has", "when", "who", "will", "no", "more", "if",
            "out", "so", "up", "said", "what", "its", "about", "into", "than", "them",
            "can", "only", "other", "new", "some", "could", "time", "very", "my", "did",
            "do", "now", "such", "like", "just", "then", "also", "after", "should", "well",
            "any", "most", "these", "two", "may", "each", "how", "many", "before", "must",
            "through", "over", "where", "much", "even", "our", "me", "back", "still", "own",
        ],
    ),
    "fr": LanguageProfile(
        code="fr",
        name="French",
        spacy_model="fr_core_news_sm",
        textstat_code="fr",
        quote_marks=["\u00ab", "\u00bb", '"', "\u201c", "\u201d"],
        contraction_suffixes=["l'", "d'", "n'", "j'", "s'", "c'", "m'", "t'", "qu'"],
        function_words=[
            "de", "la", "le", "et", "les", "des", "en", "un", "du", "une",
            "que", "est", "dans", "qui", "a", "pour", "pas", "au", "sur", "ne",
            "se", "par", "ce", "il", "sont", "plus", "avec", "son", "mais", "comme",
            "on", "tout", "nous", "sa", "ou", "lui", "y", "elle", "ses", "cette",
            "ils", "aux", "deux", "ces", "je", "leur", "bien", "aussi", "peut", "entre",
            "sans", "nos", "vous", "mon", "fait", "ont", "si", "elles", "moi", "mes",
            "ni", "me", "ma", "te", "ton", "ta", "tes", "votre", "vos", "notre",
        ],
    ),
    "de": LanguageProfile(
        code="de",
        name="German",
        spacy_model="de_core_news_sm",
        textstat_code="de",
        quote_marks=["\u201e", "\u201c", "\u00bb", "\u00ab", '"'],
        contraction_suffixes=[],  # German doesn't use contractions in writing
        function_words=[
            "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich",
            "des", "auf", "für", "ist", "im", "dem", "nicht", "ein", "eine", "als",
            "auch", "es", "an", "werden", "aus", "er", "hat", "dass", "sie", "nach",
            "wird", "bei", "einer", "um", "am", "sind", "noch", "wie", "einem", "über",
            "so", "zum", "war", "haben", "nur", "oder", "aber", "vor", "zur", "bis",
            "mehr", "durch", "man", "dann", "soll", "schon", "wenn", "sein", "keine", "ihre",
        ],
    ),
    "es": LanguageProfile(
        code="es",
        name="Spanish",
        spacy_model="es_core_news_sm",
        textstat_code="es",
        quote_marks=["\u00ab", "\u00bb", '"', "\u201c", "\u201d"],
        contraction_suffixes=[],  # Spanish has al/del but spaCy handles these
        function_words=[
            "de", "la", "que", "el", "en", "y", "a", "los", "del", "se",
            "las", "por", "un", "para", "con", "no", "una", "su", "al", "es",
            "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este", "si",
            "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta",
            "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les",
            "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mi",
        ],
    ),
    "it": LanguageProfile(
        code="it",
        name="Italian",
        spacy_model="it_core_news_sm",
        textstat_code="it",
        quote_marks=["\u00ab", "\u00bb", '"', "\u201c", "\u201d"],
        contraction_suffixes=["l'", "d'", "un'", "dell'", "nell'", "sull'", "all'"],
        function_words=[
            "di", "e", "il", "la", "che", "in", "un", "a", "per", "è",
            "una", "del", "non", "si", "i", "da", "le", "dei", "con", "al",
            "sono", "ha", "lo", "come", "suo", "ma", "più", "nel", "gli", "alla",
            "anche", "questo", "fra", "se", "o", "già", "era", "essere", "ci", "tra",
            "quando", "cui", "molto", "della", "degli", "delle", "nelle", "sulle", "dalle", "alle",
        ],
    ),
    "nl": LanguageProfile(
        code="nl",
        name="Dutch",
        spacy_model="nl_core_news_sm",
        textstat_code="nl",
        quote_marks=['"', "\u201c", "\u201d", "\u201e"],
        contraction_suffixes=[],
        function_words=[
            "de", "van", "een", "het", "en", "in", "is", "dat", "op", "te",
            "zijn", "voor", "met", "die", "niet", "hij", "was", "er", "maar", "ook",
            "als", "aan", "om", "dan", "nog", "wel", "bij", "uit", "worden", "door",
            "naar", "heeft", "hun", "dit", "haar", "wie", "wat", "al", "zo", "kan",
        ],
    ),
    "pl": LanguageProfile(
        code="pl",
        name="Polish",
        spacy_model="pl_core_news_sm",
        textstat_code="pl",
        quote_marks=["\u201e", "\u201d", '"'],
        contraction_suffixes=[],
        function_words=[
            "i", "w", "na", "z", "do", "nie", "że", "to", "się", "o",
            "jest", "jak", "co", "ale", "za", "od", "po", "tak", "go", "by",
            "już", "tego", "je", "czy", "tym", "tej", "ten", "jego", "jej", "są",
            "tylko", "który", "dla", "może", "też", "był", "jeszcze", "będzie", "tu", "więc",
        ],
    ),
    "ru": LanguageProfile(
        code="ru",
        name="Russian",
        spacy_model="ru_core_news_sm",
        textstat_code="ru",
        quote_marks=["\u00ab", "\u00bb", '"'],
        contraction_suffixes=[],
        function_words=[
            "и", "в", "не", "на", "я", "что", "он", "с", "это", "а",
            "как", "но", "его", "к", "по", "они", "она", "было", "от", "все",
            "так", "же", "бы", "за", "мы", "для", "вы", "уже", "из", "у",
            "то", "её", "их", "только", "себя", "ещё", "был", "был", "когда", "если",
        ],
    ),
    # Tier 2: spaCy + internationalizer (no textstat readability)
    "ja": LanguageProfile(
        code="ja", name="Japanese", spacy_model="ja_core_news_sm", textstat_code=None,
        quote_marks=["\u300c", "\u300d", "\u300e", "\u300f"],
        function_words=["の", "に", "は", "を", "た", "が", "で", "て", "と", "し",
                        "れ", "さ", "ある", "いる", "も", "する", "から", "な", "こと", "として"],
    ),
    "ko": LanguageProfile(
        code="ko", name="Korean", spacy_model="ko_core_news_sm", textstat_code=None,
        quote_marks=['"', "\u201c", "\u201d"],
        function_words=["이", "그", "의", "에", "는", "을", "를", "와", "과", "한",
                        "로", "도", "가", "에서", "으로", "하다", "있다", "되다", "수", "것"],
    ),
    "zh": LanguageProfile(
        code="zh", name="Chinese", spacy_model="zh_core_web_sm", textstat_code=None,
        quote_marks=["\u201c", "\u201d", "\u300c", "\u300d"],
        function_words=["的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
                        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去"],
    ),
    "pt": LanguageProfile(
        code="pt", name="Portuguese", spacy_model="pt_core_news_sm", textstat_code=None,
        quote_marks=["\u00ab", "\u00bb", '"', "\u201c", "\u201d"],
        function_words=["de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
                        "é", "com", "não", "uma", "os", "no", "se", "na", "por", "mais"],
    ),
    "sv": LanguageProfile(
        code="sv", name="Swedish", spacy_model="sv_core_news_sm", textstat_code=None,
        quote_marks=['"', "\u201d", "\u201c"],
        function_words=["och", "i", "att", "det", "som", "en", "på", "är", "av", "för",
                        "med", "har", "den", "till", "inte", "var", "jag", "han", "ett", "om"],
    ),
    "da": LanguageProfile(
        code="da", name="Danish", spacy_model="da_core_news_sm", textstat_code=None,
        quote_marks=['"', "\u201c", "\u201d"],
        function_words=["og", "i", "at", "det", "er", "en", "den", "til", "på", "for",
                        "med", "har", "som", "af", "et", "der", "var", "de", "ikke", "han"],
    ),
    "fi": LanguageProfile(
        code="fi", name="Finnish", spacy_model="fi_core_news_sm", textstat_code=None,
        quote_marks=['"', "\u201d", "\u201c"],
        function_words=["ja", "on", "ei", "se", "että", "hän", "oli", "mutta", "kun", "niin",
                        "jo", "tai", "olla", "vain", "kuin", "myös", "sen", "ne", "nyt", "ovat"],
    ),
    "el": LanguageProfile(
        code="el", name="Greek", spacy_model="el_core_news_sm", textstat_code=None,
        quote_marks=["\u00ab", "\u00bb", '"'],
        function_words=["και", "το", "η", "ο", "να", "τα", "σε", "που", "με", "δεν",
                        "για", "τη", "τον", "τις", "ένα", "από", "της", "είναι", "αυτό", "στο"],
    ),
    "ro": LanguageProfile(
        code="ro", name="Romanian", spacy_model="ro_core_news_sm", textstat_code=None,
        quote_marks=["\u201e", "\u201d", '"'],
        function_words=["de", "și", "în", "a", "la", "cu", "pe", "o", "un", "nu",
                        "este", "din", "care", "se", "că", "mai", "pentru", "sunt", "dar", "au"],
    ),
    "uk": LanguageProfile(
        code="uk", name="Ukrainian", spacy_model="uk_core_news_sm", textstat_code=None,
        quote_marks=["\u00ab", "\u00bb", '"'],
        function_words=["і", "в", "не", "на", "що", "я", "з", "він", "це", "а",
                        "як", "але", "його", "до", "по", "вони", "вона", "було", "від", "все"],
    ),
}


def get_profile(lang: str) -> LanguageProfile:
    """Get a language profile by code."""
    if lang not in PROFILES:
        available = ", ".join(sorted(PROFILES.keys()))
        raise ValueError(f"Unsupported language: {lang}. Available: {available}")
    return PROFILES[lang]

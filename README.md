<p align="center">
  <strong>Colophon</strong><br>
  <em>A writer's fingerprint, extracted by machine.</em>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#what-it-does">What It Does</a> &middot;
  <a href="#commands">Commands</a> &middot;
  <a href="#style-transfer">Style Transfer</a> &middot;
  <a href="#vector-database">Vector Database</a> &middot;
  <a href="#multilingual">Multilingual</a> &middot;
  <a href="#performance">Performance</a> &middot;
  <a href="#project-structure">Project Structure</a>
</p>

---

Colophon is a Python CLI that extracts an author's stylometric fingerprint from their writing and uses it to restyle other text in their voice. All analysis is deterministic — no ML, no LLM. The style transfer step uses any LLM provider, guided by a system prompt built entirely from the measured data.

```text
ingest (PDF/EPUB/TXT/MD) -> analyze (12 features) -> stylize (LLM rewrite)
                                                   -> report (HTML/console)
                                                   -> compare (Burrows' Delta)
                                                   -> index (pgvector storage)
                                                   -> search-style (nearest neighbor)
```

## Quick Start

Requires Python 3.10+ and pip.

```bash
git clone https://github.com/Tom-R-Main/Colophon.git
cd Colophon
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
python -m spacy download en_core_web_sm
```

Test with any text file:

```bash
colophon analyze my_book.pdf
```

That's it. You'll get a full stylometric report in the terminal and a `.analysis.json` file.

## What It Does

Colophon computes twelve deterministic features from text in 18 languages:

| Feature | What It Measures | Library |
|---------|-----------------|---------|
| **Readability** | Flesch-Kincaid, Gunning Fog, Coleman-Liau, SMOG, ARI | textstat |
| **Sentence rhythm** | Length distribution, mean/median/stdev/skewness | spaCy |
| **Vocabulary richness** | Type-token ratio, hapax legomena, Yule's K, Honore's R | spaCy + stdlib |
| **Function words** | Top 100 function word frequencies (language-specific lists) | spaCy |
| **POS distribution** | Noun/verb/adj/adv proportions, adj-noun ratio | spaCy |
| **N-grams** | Word bigrams/trigrams, character trigrams | spaCy + stdlib |
| **Punctuation** | Per-1000-word rates for commas, dashes, quotes, etc. | stdlib |
| **Contractions** | Contraction rate and types (language-aware detection) | spaCy |
| **Sentence openers** | What words/POS tags start sentences, conjunction-start rate | spaCy |
| **Paragraph structure** | Paragraph length, one-sentence paragraph ratio | spaCy |
| **Dialogue patterns** | Quoted speech ratio, attribution verbs (locale-aware quotes) | spaCy |
| **Syntax complexity** | Dependency tree depth, verb tense distribution, sentence types | spaCy |

These features are the same ones used in academic stylometry and forensic authorship attribution. They capture unconscious habits — sentence length, function word choice, punctuation tics — that are more distinctive than vocabulary or topic.

## Commands

```bash
colophon ingest <file>                    # Extract text from PDF/EPUB/TXT/MD
colophon analyze <file>                   # Run all 12 analyzers, print results
colophon compare <corpus-dir> <unknown>   # Authorship attribution (Burrows' Delta)
colophon report <analysis.json>           # Interactive HTML report (Plotly)
colophon stylize <article> <style.json>   # Rewrite article in the analyzed style
colophon index <file> --author "Name"     # Store style profile in pgvector
colophon search-style <file>              # Find similar authors/documents
```

### `ingest`

Extracts and normalizes text. Segments into articles/chapters where possible.

```bash
colophon ingest book.pdf                       # -> book.colophon.json
colophon ingest book.epub                      # EPUB chapter-aware extraction
colophon ingest book.pdf --no-segment          # Treat as one document
colophon ingest book.pdf --segment-pattern "^[A-Z]{3,}"  # Custom regex
```

Supported formats: `.pdf` (pymupdf + pdfplumber), `.epub` (ebooklib), `.txt`, `.md`

For scanned PDFs, install OCR support: `pip install colophon[ocr]` (requires system Tesseract).

### `analyze`

Runs the full stylometric pipeline. Outputs a Rich console summary and saves JSON.

```bash
colophon analyze book.pdf                          # Ingest + analyze in one step
colophon analyze book.colophon.json                # Analyze pre-ingested document
colophon analyze book.pdf --analyzers readability,sentences  # Specific analyzers
colophon analyze buch.epub --lang de               # German analysis
```

Available analyzers: `readability`, `sentences`, `vocabulary`, `function_words`, `pos`, `ngrams`, `punctuation`, `contractions`, `sentence_openers`, `paragraphs`, `dialogue`, `syntax`

Documents over 50,000 words are automatically chunked and processed in parallel across CPU cores. See [Performance](#performance).

### `compare`

Authorship attribution using Burrows' Delta. Organize a corpus directory with one subdirectory per author:

```
corpus/
  hemingway/
    sun_also_rises.txt
  royko/
    one_more_time.colophon.json
```

```bash
colophon compare corpus/ mystery_text.txt
```

Lower delta score = more similar style.

### `report`

Generates a self-contained HTML report with interactive Plotly charts.

```bash
colophon report analysis.json -o report.html
colophon report analysis.json --standalone       # Embed Plotly.js (~3MB, no CDN)
colophon report analysis.json --format json      # Raw JSON export
```

### `stylize`

Rewrites an article in the analyzed author's voice. Provider-agnostic — set one API key in `.env` and it auto-detects.

```bash
# Set any provider's API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env    # or OPENAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY

# Rewrite an article
colophon stylize article.txt royko.analysis.json

# With few-shot examples from the source text
colophon stylize article.txt royko.analysis.json -s royko.colophon.json

# Preview the system prompt (no API call)
colophon stylize article.txt royko.analysis.json --show-prompt

# Explicit provider and model
colophon stylize article.txt royko.analysis.json --provider openai --llm-model gpt-4o
```

Supported providers: Anthropic, OpenAI, Gemini, OpenRouter.

### `index`

Stores a document's style profile as a 128-dim vector in pgvector for similarity search.

```bash
colophon index book.pdf --author "Mike Royko"
colophon index book.analysis.json --author "Franz Kafka" --lang de
```

Requires a running pgvector instance. See [Vector Database](#vector-database).

### `search-style`

Finds the most stylistically similar authors and documents in the database.

```bash
colophon search-style article.txt
colophon search-style article.txt --top 5
```

## Style Transfer

The `stylize` command converts a StyleProfile into an XML-tagged system prompt with specific numeric targets:

- *"Mean sentence length: 17.7 words. Median: 15.0. Skewness: 2.93."*
- *"47% of paragraphs are a single sentence — use one-sentence paragraphs for punchlines."*
- *"12.2% of sentences start with But/And/So — start many sentences with conjunctions."*
- *"53% quoted speech — let characters speak, then comment. Attribution verb: 'said' only."*
- *"Contraction rate 22.8/1000 — always use don't, can't, it's. Never do not, cannot, it is."*

Two enhancements improve output quality:

**Few-shot examples** (`-s source.colophon.json`): Samples 5 representative passages from the source text, scored by proximity to the profile's median characteristics. These go into the prompt as `<examples>` blocks.

**Fingerprint diff**: Before sending to the LLM, Colophon analyzes the input article and computes what's different from the target style. The prompt gets specific instructions like *"SHORTEN SENTENCES: input averages 24.3, target is 17.7"* or *"ADD CONTRACTIONS: input uses 0/1000, target uses 22.8/1000."*

## Vector Database

Colophon can store style profiles in PostgreSQL with pgvector for nearest-neighbor style search.

### Setup

```bash
# Start pgvector
docker compose up -d

# Index some authors
colophon index royko.colophon.json --author "Mike Royko"
colophon index kafka.colophon.json --author "Franz Kafka" --lang de

# Search
colophon search-style unknown_article.txt
```

The default connection is `postgresql://colophon:colophon_dev@localhost:5433/colophon`. Override with `--db`.

### How It Works

Each document's StyleProfile is flattened into a 128-dimensional vector where every dimension is an interpretable stylometric feature (readability scores, sentence stats, POS proportions, punctuation rates, etc.). Vectors are log-scaled and L2-normalized so all features contribute equally to similarity regardless of their natural scale.

The database uses HNSW indexes for approximate nearest-neighbor search — fast even across thousands of indexed authors.

### Schema

- `style_profiles` — per-document vectors + full JSON features
- `author_profiles` — aggregated mean vectors across all documents by an author

## Multilingual

Colophon supports 18 languages with language-specific function word lists, contraction detection, and quote mark conventions.

**Tier 1** — full readability + all analyzers:

| Language | Code | spaCy Model |
|----------|------|-------------|
| English | `en` | `en_core_web_sm` |
| French | `fr` | `fr_core_news_sm` |
| German | `de` | `de_core_news_sm` |
| Spanish | `es` | `es_core_news_sm` |
| Italian | `it` | `it_core_news_sm` |
| Dutch | `nl` | `nl_core_news_sm` |
| Polish | `pl` | `pl_core_news_sm` |
| Russian | `ru` | `ru_core_news_sm` |

**Tier 2** — all analyzers except readability:

`ja`, `ko`, `zh`, `pt`, `sv`, `da`, `fi`, `el`, `ro`, `uk`

```bash
# Analyze Kafka in German
python -m spacy download de_core_news_sm
colophon analyze kafka.epub --lang de

# Analyze Tolstoy in Russian
python -m spacy download ru_core_news_sm
colophon analyze tolstoy.epub --lang ru
```

## Performance

For large documents (>50,000 words), Colophon automatically splits text at paragraph boundaries and processes chunks in parallel across CPU cores.

| Document | Words | Single-thread | Chunked | Speedup |
|----------|-------|--------------|---------|---------|
| Royko | 108K | ~2 min | ~28 sec | 4.3x |
| Tolstoy | 200K | ~6 min | ~80 sec | 4.7x |

Analyzers that don't need spaCy (readability, punctuation) run on the full text directly. spaCy-dependent analyzers are chunked and distributed across a process pool. Results are merged with correct statistical recomputation.

## Installation Options

Colophon uses optional dependency groups so you only install what you need:

```bash
pip install -e "."               # Core only (CLI + models, ~3 deps)
pip install -e ".[ingest]"       # + PDF, EPUB parsing
pip install -e ".[analysis]"     # + spaCy, textstat, NLTK
pip install -e ".[report]"       # + Plotly, Jinja2
pip install -e ".[stylize]"      # + Anthropic SDK, python-dotenv
pip install -e ".[embeddings]"   # + pgvector, SQLAlchemy, psycopg2
pip install -e ".[all]"          # Everything above
pip install -e ".[ocr]"          # + OCRmyPDF (needs system Tesseract)
pip install -e ".[neural]"       # + sentence-transformers, torch (future)
pip install -e ".[dev]"          # All + pytest, ruff, pyright
```

After installing `[analysis]`, download the spaCy model for your language:

```bash
python -m spacy download en_core_web_sm    # English
python -m spacy download de_core_news_sm   # German
python -m spacy download fr_core_news_sm   # French
# etc.
```

## Project Structure

```text
colophon/
├── src/colophon/
│   ├── cli.py                    # Typer CLI (7 commands)
│   ├── ingestion/                # PDF, EPUB, TXT/MD extraction + normalization
│   ├── analysis/                 # 12 deterministic analyzers + chunked pipeline
│   │   ├── pipeline.py           # Orchestrator (auto-chunks large documents)
│   │   ├── chunked.py            # Multiprocessing worker + chunk splitting
│   │   └── merge.py              # Merge partial results across chunks
│   ├── comparison/               # Burrows' Delta authorship attribution
│   ├── reporting/                # Rich console, Plotly HTML, JSON export
│   ├── stylize/                  # Style transfer
│   │   ├── prompt.py             # StyleProfile -> XML system prompt
│   │   ├── examples.py           # Few-shot passage selection
│   │   ├── diff.py               # Fingerprint delta computation
│   │   └── providers/            # Anthropic, OpenAI, Gemini, OpenRouter
│   ├── embeddings/               # Style vectorization
│   │   └── vectorize.py          # StyleProfile -> 128-dim normalized vector
│   ├── db/                       # pgvector storage layer
│   │   ├── schema.py             # SQLAlchemy models
│   │   └── operations.py         # Index, search, aggregate
│   ├── lang/                     # 18 language profiles
│   └── models/                   # Pydantic models (Document, StyleProfile)
├── docker-compose.yml            # pgvector container
├── init.sql                      # Database schema + HNSW indexes
├── pyproject.toml                # Hatchling build, optional dep groups
└── .env                          # API keys (gitignored)
```

## License

MIT

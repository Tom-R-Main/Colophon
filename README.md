<p align="center">
  <strong>Colophon</strong><br>
  <em>A writer's fingerprint, extracted by machine.</em>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#what-it-does">What It Does</a> &middot;
  <a href="#commands">Commands</a> &middot;
  <a href="#the-pipeline">The Pipeline</a> &middot;
  <a href="#style-transfer">Style Transfer</a> &middot;
  <a href="#project-structure">Project Structure</a>
</p>

---

Colophon is a Python CLI that extracts an author's stylometric fingerprint from their writing and uses it to restyle other text in their voice. All analysis is deterministic — no ML, no LLM. The style transfer step uses Claude, guided by a system prompt built entirely from the measured data.

```text
ingest (PDF/EPUB/TXT/MD) -> analyze (12 features) -> stylize (Claude rewrite)
                                                  -> report (HTML/console)
                                                  -> compare (Burrows' Delta)
```

## Quick Start

Requires Python 3.10+ and pip.

```bash
git clone https://github.com/Tom-R-Main/colophon.git
cd colophon
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

Colophon computes twelve deterministic features from any English text:

| Feature | What It Measures | Library |
|---------|-----------------|---------|
| **Readability** | Flesch-Kincaid, Gunning Fog, Coleman-Liau, SMOG, ARI | textstat |
| **Sentence rhythm** | Length distribution, mean/median/stdev/skewness | spaCy |
| **Vocabulary richness** | Type-token ratio, hapax legomena, Yule's K, Honore's R | spaCy + stdlib |
| **Function words** | Top 100 function word frequencies (Burrows 1987 list) | spaCy |
| **POS distribution** | Noun/verb/adj/adv proportions, adj-noun ratio | spaCy |
| **N-grams** | Word bigrams/trigrams, character trigrams | spaCy + stdlib |
| **Punctuation** | Per-1000-word rates for commas, dashes, quotes, etc. | stdlib |
| **Contractions** | Contraction rate and types (don't, it's, etc.) | spaCy |
| **Sentence openers** | What words/POS tags start sentences, conjunction-start rate | spaCy |
| **Paragraph structure** | Paragraph length, one-sentence paragraph ratio | spaCy |
| **Dialogue patterns** | Quoted speech ratio, attribution verbs (said vs. exclaimed) | spaCy |
| **Syntax complexity** | Dependency tree depth, verb tense distribution, sentence types | spaCy |

These features are the same ones used in academic stylometry and forensic authorship attribution. They capture unconscious habits — sentence length, function word choice, punctuation tics — that are more distinctive than vocabulary or topic.

## Commands

```bash
colophon ingest <file>                    # Extract text from PDF/EPUB/TXT/MD
colophon analyze <file>                   # Run all 7 analyzers, print results
colophon compare <corpus-dir> <unknown>   # Authorship attribution (Burrows' Delta)
colophon report <analysis.json> -o out.html  # Interactive HTML report (Plotly)
colophon stylize <article> <style.json>   # Rewrite article in the analyzed style
```

### `ingest`

Extracts and normalizes text. Segments into articles/chapters where possible.

```bash
colophon ingest book.pdf                       # -> book.colophon.json
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
colophon analyze book.pdf --analyzers readability,sentences  # Run specific analyzers
```

Available analyzers: `readability`, `sentences`, `vocabulary`, `function_words`, `pos`, `ngrams`, `punctuation`, `contractions`, `sentence_openers`, `paragraphs`, `dialogue`, `syntax`

### `compare`

Authorship attribution using Burrows' Delta. Organize a corpus directory with one subdirectory per author:

```
corpus/
  hemingway/
    sun_also_rises.txt
    old_man_and_sea.txt
  fitzgerald/
    gatsby.txt
  royko/
    one_more_time.colophon.json
```

```bash
colophon compare corpus/ mystery_text.txt
```

Lower delta score = more similar style. The algorithm computes z-scored word frequency vectors and measures mean absolute deviation.

### `report`

Generates a self-contained HTML report with interactive Plotly charts.

```bash
colophon report analysis.json -o report.html
colophon report analysis.json --standalone       # Embed Plotly.js (~3MB, no CDN)
colophon report analysis.json --format json      # Raw JSON export
```

### `stylize`

Rewrites an article in the analyzed author's voice. Requires an Anthropic API key.

```bash
# Set your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# Rewrite an article in Royko's style
colophon stylize article.txt royko.analysis.json

# Preview the generated system prompt (no API call)
colophon stylize article.txt royko.analysis.json --show-prompt

# Use a different model
colophon stylize article.txt royko.analysis.json --llm-model claude-opus-4-20250514
```

The system prompt is built entirely from the StyleProfile data — sentence length targets, vocabulary register, punctuation habits, function word frequencies, POS ratios. Every instruction is backed by a measured number.

## The Pipeline

```text
┌──────────┐     ┌──────────┐     ┌──────────┐
│  ingest  │────>│ analyze  │────>│  report  │
│ PDF/EPUB │     │ 7 feats  │     │ HTML/CLI │
│ TXT/MD   │     │          │     │          │
└──────────┘     └────┬─────┘     └──────────┘
                      │
                      ├──────────>┌──────────┐
                      │           │ compare  │
                      │           │ Delta    │
                      │           └──────────┘
                      │
                      └──────────>┌──────────┐
                                  │ stylize  │
                                  │ Claude   │
                                  └──────────┘
```

**Ingest** normalizes text from any supported format into a `Document` model with segments. **Analyze** runs deterministic feature extraction. The resulting `StyleProfile` feeds three outputs: visual **reports**, authorship **comparison**, and AI **style transfer**.

## Installation Options

Colophon uses optional dependency groups so you only install what you need:

```bash
pip install -e "."               # Core only (CLI + models, ~3 deps)
pip install -e ".[ingest]"       # + PDF, EPUB parsing
pip install -e ".[analysis]"     # + spaCy, textstat, NLTK
pip install -e ".[report]"       # + Plotly, Jinja2
pip install -e ".[stylize]"      # + Anthropic SDK
pip install -e ".[all]"          # Everything except OCR
pip install -e ".[ocr]"          # + OCRmyPDF (needs system Tesseract)
pip install -e ".[dev]"          # All + pytest, ruff, pyright
```

After installing `[analysis]`, download the spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Project Structure

```text
colophon/
├── src/colophon/
│   ├── cli.py                    # Typer CLI (ingest, analyze, compare, report, stylize)
│   ├── ingestion/                # PDF, EPUB, TXT/MD extraction + normalization
│   ├── analysis/                 # 7 deterministic analyzers + pipeline orchestrator
│   ├── comparison/               # Burrows' Delta (vendored, no faststylometry dep)
│   ├── reporting/                # Rich console, Plotly HTML, JSON export
│   │   └── templates/report.html # Jinja2 template for HTML reports
│   ├── stylize/                  # System prompt builder + Anthropic client
│   └── models/                   # Pydantic models (Document, StyleProfile, etc.)
├── assets/                       # Test data (gitignored)
├── pyproject.toml                # Hatchling build, optional dep groups
└── .env                          # API keys (gitignored)
```

## How Style Transfer Works

1. `analyze` computes a `StyleProfile` across 12 dimensions — readability, sentence rhythm, vocabulary, function words, POS ratios, punctuation, contractions, sentence openers, paragraph structure, dialogue patterns, and syntactic complexity
2. `stylize` converts that profile into an XML-tagged system prompt with specific numeric targets:
   - *"Mean sentence length: 17.7 words. Median: 15.0. Skewness: 2.93."*
   - *"47% of paragraphs are a single sentence — use one-sentence paragraphs for punchlines."*
   - *"12.2% of sentences start with But/And/So — start many sentences with conjunctions."*
   - *"53% quoted speech — let characters speak, then comment. Attribution verb: 'said' only."*
   - *"Contraction rate 22.8/1000 — always use don't, can't, it's. Never do not, cannot, it is."*
3. Claude rewrites the input article to match those targets while preserving all original facts and quotes

Use `--show-prompt` to inspect the generated system prompt without making an API call.

## License

MIT

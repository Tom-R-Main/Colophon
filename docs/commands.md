# Command reference

Use `colophon --help`, `colophon voice --help`, or `colophon COMMAND --help` for the exact installed arguments and defaults. The CLI owns option syntax; this page helps you choose a route.

## Voice profiles

| Command | Input and result |
|---|---|
| `doctor` | Reports package/model presence without installing anything or reading credentials |
| `corpus validate` | Checks a local manifest, source ownership, independent works and split separation |
| `voice extract` | Builds an author profile from included training works, with optional annotations/corrections |
| `voice inspect` | Reads a profile; `--evidence` includes supporting source passages |
| `voice export` | Exports JSON or Markdown; `--all-evidence` expands Markdown evidence |
| `voice prompt` | Exports one of four prompt modes; `--task generate` prepares new-writing guidance; `--receipt` requires `-o` and checks saved bytes |
| `voice evaluate-profile` | Compares one fixed profile with unseen pieces; reports stability, not authorship probability |
| `voice evaluate` | Runs offline retrieval baselines on a multi-author held-out corpus; does not generate writing |
| `voice evaluate-writing` | Scores supplied JSON/JSONL outputs, retaining separate channels and unscored/error states |

The [voice guide](voice-toolchain.md) owns corpus fields, quotation policy, corrections and evaluation formats. The [agent guide](agent-usage.md) owns receipt interpretation and recovery.

## Individual documents

```bash
colophon ingest corpus/article.html -o outputs/article.colophon.json
colophon analyze outputs/article.colophon.json -o outputs/article.analysis.json
colophon report outputs/article.analysis.json -o outputs/article.report.html
```

Ingestion supports PDF, EPUB, HTML/HTM, TXT and Markdown. Install `.[ingest,analysis]` and the corresponding spaCy model. Plain text intake needs no PDF/EPUB parser. Scanned PDF OCR is an optional `ocr` extra and requires system Tesseract.

Analysis has twelve feature families. A feature can be measured, unavailable or inapplicable; unavailable does not mean zero. English is the current checked path. The former chunking options now use exact whole-document analysis because merging chunks changed measurements. Large-document capacity is not qualified.

Reporting uses the `report` extra. `--standalone` embeds Plotly rather than loading it from a CDN. `compare` accepts a directory containing one subdirectory per author and a candidate document; its Burrows' Delta distance is a similarity measure, not a probability.

## Optional integrations

These routes use separate services or provider credentials. They are not needed for profile extraction or prompt export.

| Route | Setup and boundary |
|---|---|
| `stylize ARTICLE STYLE` | Uses a single-document `.analysis.json` and a configured Anthropic, OpenAI, Gemini or OpenRouter provider. `--show-prompt` previews without generating. Optional source examples use `--source`. |
| `index FILE --author NAME` | Stores a document/style vector in PostgreSQL with pgvector; requires the `embeddings` extra and a configured database |
| `search-style FILE` | Queries that database for nearby document/author vectors |
| `serve` | Starts the local FastAPI UI; requires the `web` extra plus the dependencies of the operations you use |

Install the relevant extras declared in [pyproject.toml](../pyproject.toml), or `.[all]` for the combined runtime dependencies. A writing provider still needs separate credentials. Keep actual credentials in the process environment or an ignored local environment file; never add them to examples or commits. Use explicit `--provider` and `--llm-model` values when an experiment needs a fixed writing setup.

The legacy `scripts/build_corpus.py` downloads texts and populates a database. It is an optional local workflow, not the synthetic first run. Stored corpus profiles are not distributed with this checkout. Check source rights and dependencies before using it.

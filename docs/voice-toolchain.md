# Extract and inspect a writer's voice

Colophon measures a corpus and records the passages behind its observations. Its voice profile is a reusable JSON artifact: source identities, contributor context, per-document measurements, grounded traits, counterexamples, exemplars and writer corrections. It can be exported to Markdown or a prompt for another writing tool.

The first implementation uses numerical analysis and conservative structural rules. English rules locate explicit contrast/cause markers, pronouns and uncertainty markers; they do not establish a writer's intention or personality. Openings, endings and short paragraphs are described only where source boundaries are usable. More nuanced interpretations can be supplied as annotations with exact source spans. Valid citations establish location, not the truth of an interpretation.

## Install

Python 3.10+ is declared by the package. The pinned development environment and current local verification use Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/colophon doctor
```

For a smaller runtime installation, use `pip install -e ".[analysis,ingest]"` and install the English model. TXT/MD extraction does not need the `ingest` extra. Providers, PostgreSQL and model API keys are not needed for the voice commands below. Model downloads are an explicit installation step, never part of corpus loading or extraction.

## Prepare independent source pieces

Keep a JSON manifest beside its source files. Each entry names the writer whose language is being measured, its original work, language, genre, topic, date if known, and an extraction or evaluation split.

```json
{
  "schema_version": "1",
  "name": "Writer's columns",
  "documents": [
    {
      "id": "column-01",
      "path": "columns/column-01.txt",
      "author": "writer-id",
      "work_id": "original-column-01",
      "language": "en",
      "genre": "column",
      "topic": "local-government",
      "date": "2026-09-01",
      "split": "train"
    }
  ]
}
```

Use `train` pieces to build profiles. Reserve `dev` for threshold or method choices and `test` for the final comparison. Revisions, excerpts and chunks of one original piece belong to the same `work_id` and split. Profile construction currently requires one independent work per selected document; split a collection into original articles, rather than presenting chapters as independent works.

Sources must remain inside the manifest directory. TXT, MD, HTML, EPUB, PDF and existing `.colophon.json` documents are accepted. Exact duplicates are rejected. Cross-split near-duplicate/excerpt overlap is checked using five-word shingles and conservative containment thresholds; that check cannot guarantee the absence of every paraphrased or short-text leak.

Additional fields:

| Field | Purpose |
|---|---|
| `include: false`, `exclusion_reason` | Keep an exclusion decision in the manifest without analyzing that source |
| `contributors` | List `name` and role: `original_author`, `translator`, `editor` or `quoted` |
| `voice_author` | Assign the linguistic surface to its actual writer; translated text must identify a listed translator as the voice owner |
| `excluded_spans` | List `start`, `end` and `reason` for editorial or other non-author material; offsets refer to normalized document text |
| `paragraphs_verified` | Mark reviewed PDF block boundaries usable for paragraph measurement; this is a human declaration, not a parser quality score |
| `quotation_policy` | `exclude_all` pairs inline quotation marks conservatively; `explicit_only` keeps inline quoted terms while excluding declared non-author blocks/spans |

By default, paired double/guillemet and selected East Asian quotation spans are excluded from author measurements. Apostrophes are not treated as speech delimiters. Headings, furniture, quotation blocks and non-author-attributed blocks are excluded first, so quotes in those regions cannot consume later author prose. Unclosed author quotes are conservatively excluded through the end and flagged. For a reviewed corpus, `explicit_only` preserves the writer's own quoted labels; attributed speech then needs explicit block ownership or span exclusions. Unmarked quotations, complex nesting, OCR errors or editorial changes may still require manual review.

EPUB paragraphs and headings retain their source item and exact normalized offsets. PDF items retain page/region provenance, but physical blocks are not automatically certified prose paragraphs. PDF paragraph statistics and structural traits abstain until reviewed. Exact offsets identify normalized text; they are not offsets into raw PDF bytes.

HTML ingestion preserves logical blocks and DOM locators without running scripts or fetching resources. It recognizes semantic article roots and a legacy font/BR layout. Navigation, notes, promotional regions, attributed epigraphs and recognizable code/data regions are separate from prose. These are structure-based heuristics, so inspect unusual pages and the recorded warnings.

```bash
colophon ingest column.pdf -o column.colophon.json
colophon corpus validate corpus.json
```

Inspect the ingested text, blocks and warnings before adding exclusions. The corpus loader retains both raw-source and normalized-text SHA-256 identities and masks excluded text without shifting normalized offsets.

## Extract, inspect and reuse

```bash
colophon voice extract corpus.json --author writer-id -o writer.voice.json
colophon voice inspect writer.voice.json
colophon voice inspect writer.voice.json --evidence
colophon voice export writer.voice.json --format markdown -o writer.voice.md
colophon voice export writer.voice.json --all-evidence -o writer.complete.md
colophon voice prompt writer.voice.json --mode hybrid -o writer.prompt.txt
```

Prompt modes are `numeric`, `exemplars`, `qualitative` and `hybrid`. Exporting a prompt does not send corpus text to a provider. Numerical ranges describe variation; they are not quotas. Quoted facts and experiences must not be invented to match a style.

`voice prompt writer.voice.json --task generate --mode hybrid` exports guidance for a new piece from supplied facts. The default `--task rewrite` preserves the existing rewrite prompt. All generation modes share content constraints: reasoning and clearly hypothetical examples are allowed; invented empirical claims, reporting, experiences and new beliefs attributed to the source writer are forbidden. Numeric generation guidance uses twelve fixed features and equal-document medians with inclusive Q25–Q75 intervals from saved, available training measurements, omitting full extrema and length-sensitive vocabulary ratios. Qualitative generation guidance includes tentative/accepted interpretations and writer replacements, omits automatic literal-marker observations, and extracts an existing `Inferred guidance:` portion when present. Writer replacements, scope exceptions and evidence references remain visible. Exemplars are unchanged; the full JSON remains the canonical evidence record. These exports are experimental guidance, not evidence of generated voice quality.

Markdown bounds the displayed sources and evidence with explicit counts; JSON keeps the full record. Source excerpts are selected deterministically across available genres, topics and publication eras, alternating eligible opening/middle/ending positions. This improves context coverage without claiming the passages are stylistically optimal.

Omit `-o` for machine-readable JSON or exported prose on stdout. Failures return a nonzero exit status and an error on stderr. Artifact writes replace a completed temporary file, so an extraction failure leaves an existing output intact.

Profiles weight independent pieces equally and retain each document's measurements. They record measurement/schema/model versions, language and feature availability. Missing features stay distinct from measured zero. Recurrent marker observations have support and counterexample spans; one-piece observations remain tentative. No numerical cutoff establishes universal sample sufficiency.

The profile ID hashes included sources, ownership/exclusion configuration, blocks, extraction metadata, analysis/model versions, annotations and corrections. A changed timestamp alone does not change identity. It is a content/configuration identity, not a signature or proof of literary validity.

## Add interpretations and writer corrections

The writer can accept, reject or replace a generated trait. Find its ID with `voice inspect`, then supply a JSON list:

```json
[
  {"trait_id": "short-ending", "action": "reject", "note": "This only describes the selected columns."},
  {"trait_id": "first-person", "action": "replace", "statement": "Use first person in personal columns when the brief supports it.", "note": "Writer's preference"}
]
```

```bash
colophon voice extract corpus.json --author writer-id --corrections corrections.json -o writer.corrected.json
```

Replacements are marked `writer_supplied` and retain the original observation. They are preferences, not newly verified source claims. Rejected traits remain inspectable and are omitted from writing guidance.

`--annotations annotations.json` accepts a list of `VoiceTrait` records. Each needs a unique `id`, category (`openings`, `endings`, `argument`, `stance`, `rhythm`, `transitions` or `quotations`), `statement`, and supporting `evidence`. Every evidence record must include `document_id`, `source_hash` (the normalized full-text hash), `block_id`, `start`, `end`, `text` and `role`. Hashes, text, offsets and author ownership are checked against included training sources; held-out, stale or excluded passages are rejected. Counterexamples and scope exceptions can be included. Imported annotations are always tentative, even if their input claims a stronger status.

## Evaluate on unseen pieces

```bash
colophon voice evaluate corpus.json --split test -o retrieval.json
```

The offline comparison requires at least two training authors, one compatible language and held-out documents. It fits vocabulary, scaling and centroids on training data only:

| Method | What is measured |
|---|---|
| `numerical` | Distance to equal-document author centroids of measured features |
| `exemplars_only` | Nearest training-document character-trigram distance; a topic-sensitive retrieval proxy |
| `qualitative` | Distance between literal rule-marker/structure features; no LLM judgment |
| `hybrid` | Equal-weight numerical and rule-marker distances |

These are retrieval baselines, not four generated-writing experiments. The report records the training fit/hash, per-case topic/genre, rankings, coverage, errors and per-author results. Unknown-author rejection is unavailable without separate known-author and unknown-author development cases. A nearest neighbor is not an authorship probability. No confidence interval is fabricated from a few correlated samples.

For a one-author corpus, validate the saved profile against unseen pieces directly:

```bash
colophon voice evaluate-profile writer.voice.json corpus.json --split test -o stability.json
```

This binds the fixed training sources, blocks, exclusions and model versions; reports feature distributions, drift and missing/error coverage; and rechecks literal rule recurrence. Interpretive annotations remain `requires_semantic_review`. Same-author stability is useful evidence about scope, but does not establish authorship discrimination or recognizable generated voice.

To compare actual generated or human-written outputs, export the four prompt modes, produce outputs using a controlled provider/model setup, and retain them as JSON or JSONL cases. Generation is separate and requires its own provider configuration. Cases contain unique `id`, `variant`, `input`, `output`, and optional `references`, `exemplars` and `human` ratings (`voice`, `naturalness`, `factuality`, each between 0 and 1).

```bash
colophon voice evaluate-writing writing-cases.jsonl -o writing-results.json
```

Results retain independent channels for voice, number/quotation preservation flags, copying, naturalness and factuality. Missing human ratings are unscored. Without a human voice rating, reference character-trigram overlap is labeled a topic-sensitive proxy. Copying checks report exact shared passages and eight-word overlap; necessary quotations and common phrases need interpretation. Number/quote matches do not prove factual equivalence.

The Python `run_scorers` boundary supports additional named graders. Each gets an isolated case; a failure or mutation cannot erase another channel's evidence. Coverage retains scored, unscored and error counts independently of grader order. No favorable aggregate can hide a missing factuality judgment.

## Run the demonstration and checks

```bash
colophon voice extract examples/voice-corpus/corpus.json --author "Rowan (synthetic)" -o rowan.voice.json
colophon voice export rowan.voice.json -o rowan.voice.md
colophon voice evaluate examples/voice-corpus/corpus.json -o retrieval.json
colophon voice evaluate-writing examples/voice-corpus/writing-cases.json -o writing.json
pytest -q
ruff check src/ tests/
pyright
```

The fixtures exercise behavior and failure controls. They cannot establish that a profile captures a real writer's voice. The next quality test needs permissioned real pieces across topics, independent works held out, blinded writer/reader ratings, and separate preservation/copying assessment.

## Compatibility and limits

Corrected document measurements use schema/analyzer version `2` and vector schema `classical-2`. Legacy analysis files remain readable for inspection, but vectorization requires re-analysis. Stored search vectors and author aggregates are isolated by language, measurement version, vector schema and coverage. Re-analyze and re-index existing sources deliberately; no production migration runs automatically.

The old chunk arguments now select exact whole-document analysis. This preserves measurements but sacrifices parallelism; large-document resource limits have not been qualified. Contraction disambiguation depends on the linguistic model. English is the current verified NLP path; marker rules do not establish humor, beliefs, personal identity or topic-independent voice. PostgreSQL runtime, live provider behavior and real-writer quality are separate from the local CLI checks.

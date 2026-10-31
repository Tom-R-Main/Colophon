# Verify the voice toolchain

Use the project Python 3.12 environment and English spaCy model described in `voice-toolchain.md`. Tests and CLI fixtures run locally without a provider, credentials or an existing database. Temporary test databases use SQLite type adapters; PostgreSQL search statements are compiled separately.

## Required checks

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src/ tests/
.venv/bin/pyright
python3 scripts/check_publication.py --worktree
git diff --check
```

Drive the actual public commands and read back their artifacts:

```bash
.venv/bin/colophon doctor
.venv/bin/colophon corpus validate examples/voice-corpus/corpus.json
.venv/bin/colophon voice extract examples/voice-corpus/corpus.json --author "Rowan (synthetic)" -o outputs/implementation/voice-toolchain/rowan.voice.json
.venv/bin/colophon voice inspect outputs/implementation/voice-toolchain/rowan.voice.json --evidence
.venv/bin/colophon voice export outputs/implementation/voice-toolchain/rowan.voice.json -o outputs/implementation/voice-toolchain/rowan.voice.md
.venv/bin/colophon voice prompt outputs/implementation/voice-toolchain/rowan.voice.json --mode hybrid -o outputs/implementation/voice-toolchain/rowan.prompt.txt
.venv/bin/colophon voice evaluate examples/voice-corpus/corpus.json -o outputs/implementation/voice-toolchain/retrieval.json
.venv/bin/colophon voice evaluate-writing examples/voice-corpus/writing-cases.json -o outputs/implementation/voice-toolchain/writing.json
```

## Feature map and independent expectations

| Promise | Boundary and decisive expectation | Coverage |
|---|---|---|
| Local named profiles | Read-only explicit registry; external or untracked ignored files only; resolve validates `VoiceProfile` and hashes parsed bytes; discovery emits no source text; receipt identity matches selection | `tests/test_profile_registry.py`, `tests/test_profile_registry_cli.py` |
| Documented first run | Execute the README's marked commands in isolation; inspect training IDs, held-out exclusions, report evidence and exact receipt hashes; all public local links resolve | `tests/test_readme.py` |
| Publication boundary | Forced-added private/corpus/output paths fail before bytes are read; staged/CI scans report paths without disclosing matched values; synthetic examples remain eligible | `tests/test_publication.py` and `scripts/check_publication.py --tracked` |
| Correct measurements | Different function-word identities yield different vectors; punctuation alone preserves lexical denominators; three short paragraphs remain three; chunk options preserve measurements | `tests/test_measurements.py` uses real spaCy documents and independent Delta algebra |
| Descriptive guidance | Numerical values cannot produce class/comedy claims; short exemplars survive and long ones obey bounds | `tests/test_style_guidance.py` |
| Source ownership and boundaries | TXT/EPUB paragraphs match source structure; nested list/paragraph prose appears once; PDF layout stays unverified; quote/editorial exclusions preserve exact normalized offsets; duplicate author prose cannot hide behind excluded material | `tests/test_corpus_ingestion.py`, `tests/test_epub_logical_blocks.py`, `tests/test_author_view_leakage.py` and evaluation leakage tests |
| Grounded profile | Only selected training works contribute; cited text/hash/block/offsets match included author prose; mixed quotation fragments cannot become paragraphs; writer replacements retain provenance | `tests/test_voice_traits.py` and public CLI readback |
| Generation prompt exports | Four modes share factual constraints while retaining separate style channels; saved training measurements yield inclusive central intervals despite a measured paragraph outlier; singleton/unavailable/nonfinite values are safe; interpretive scope and writer corrections survive; default rewrite output is unchanged | `tests/test_voice_traits.py` and `--task generate` artifact readback in `tests/test_voice_cli.py`; no provider calls |
| Held-out comparison | Test content cannot change the fitted training vocabulary/scaler/hash; unknown-author thresholds need separate dev data; unavailable features stay unscored | `tests/test_voice_evaluation.py` |
| One-author stability | Frozen training identity stays fixed; held-out drift and missing features remain visible; interpretive annotations require semantic review | `tests/test_profile_validation.py` and `voice evaluate-profile` CLI readback |
| HTML ownership | BR paragraphs survive; linked banners, epigraphs, tables, code, notes and revised dates cannot become author openings or sentence fragments | `tests/test_html_ingestion.py`, including actual archived-page checks when local sources are available |
| Separate output judgments | A changed number, copied passage and missing human rating remain different channel outcomes; a crashing/mutating grader cannot alter other channels or completion counts | `tests/test_voice_evaluation.py` and `writing-cases.json` |
| Versioned storage | Old or incompatible vectors stay out of search SQL; updates/reassignments/deleting the last document update aggregates atomically; forced aggregate failure rolls back | `tests/test_db_profiles.py`: SQLite lifecycle plus PostgreSQL SQL compilation |
| Existing consumers | Generation task framing remains generation; truncated provider output fails instead of being saved as complete | `tests/test_provider_boundaries.py`: mocked provider transport, no live model claim |

The CLI demonstration is synthetic. It verifies artifact production, evidence handling and evaluation mechanics. Real-writer recognizability, semantic factual equivalence, large-document capacity, other linguistic models, live provider behavior, hosted CI and PostgreSQL vector execution require separate evidence. Do not turn a green local suite into those claims.

## Recovery and compatibility

An invalid corpus or failed extraction exits nonzero before replacing the output artifact. Correct source ownership, exclusions, split grouping or model readiness, then rerun the same command. Re-analysis is required for `legacy-1` profiles before vectorization. No automatic production data migration or model/provider call is part of verification.

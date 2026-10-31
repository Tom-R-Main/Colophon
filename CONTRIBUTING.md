# Contributing

Use Python 3.12 for the pinned development environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m spacy download en_core_web_sm
```

Run the checks before opening a pull request:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src/ tests/ scripts/check_publication.py
.venv/bin/pyright
git diff --check
python3 scripts/check_publication.py --worktree
```

The suite includes the README's marked quick start, executed with synthetic sources in an isolated directory. It checks saved profile, evidence and prompt receipts. It does not call a writing provider, certify literary fidelity or exercise live PostgreSQL. See [verification](docs/verification.md) for the behavior map and remaining limits.

## Documentation ownership

Keep the README short enough to choose a task and complete a first run. Put corpus preparation and interpretation in the [voice guide](docs/voice-toolchain.md), receipt/recovery details in the [agent guide](docs/agent-usage.md), and route selection in the [command reference](docs/commands.md). Exact option syntax belongs to the installed CLI's `--help`.

When a command changes, update its owning guide and run the relevant public CLI test. The README's `test:quickstart` block is executable test input; retain the markers and update its artifact assertions when the promised workflow changes. Installing packages or rendering Markdown is not proof that the commands work.

This division follows mechanisms inspected in uv and Inspect AI: route from the README to a task, make the task's output observable, and state what an example test actually checks. We keep a small CLI smoke test rather than importing a documentation generator or evaluation service. Source observations are recorded in the [documentation notes](docs/documentation-notes.md).

## Publication boundary

Commit code, documentation and small synthetic fixtures. Keep source books, essays, extracted text, real-author profiles/prompts, annotations, generated writing and experiment records under ignored `corpus/` or `outputs/`. Generated profiles and prompts can retain substantial source passages. The repository's code license does not cover those passages.

Only `examples/voice-corpus/` is the public text-fixture directory. Mark invented writers and sources as synthetic. Use invented examples in tests; never make CI depend on a developer's private library or home directory.

Before committing, inspect exactly what is staged:

```bash
git diff --cached --stat
git diff --cached --check
python3 scripts/check_publication.py
```

The checker reads index blobs, rejects forbidden paths before reading their contents, and reports paths/line numbers without printing matched secrets. `--tracked` audits the full proposed tree and is run in CI. `--worktree` also examines nonignored local candidates. Optional `--source-overlap DIR` compares distinctive twenty-word runs against local sources; keep those sources ignored. None of these checks grants publication rights or proves that every secret or copied passage is absent.

Ignore rules don't affect already tracked files. If an artifact was tracked, remove it from the index while preserving the local file, review that deletion, and rerun the tree check. Do not force-add private artifacts to make a test pass. Do not put private extracts, machine paths or credentials into PR bodies, review comments or logs.

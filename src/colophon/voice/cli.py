"""Public local CLI for corpus validation, voice profiles and offline evaluation."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import typer

from colophon.voice.models import TraitCorrection, VoiceProfile, VoiceTrait

app = typer.Typer(help="Extract, inspect and evaluate evidence-linked author voice profiles.", no_args_is_help=True)
corpus_app = typer.Typer(help="Validate local corpus sources and evaluation splits.", no_args_is_help=True)


def _write(value: str, output: Path | None) -> None:
    if output is None:
        typer.echo(value.rstrip())
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=output.parent, prefix=f".{output.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(value)
            if not value.endswith("\n"):
                file.write("\n")
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _json(value: object, output: Path | None) -> None:
    _write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), output)


def _fail(error: Exception) -> None:
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(1)


@corpus_app.command("validate")
def validate(manifest: Path) -> None:
    """Read a JSON manifest, check source ownership, duplicates and split leakage."""
    from colophon.voice.manifest import load_corpus

    try:
        loaded = load_corpus(manifest)
        _json(
            {
                "valid": True,
                "included_documents": len(loaded),
                "documents": [
                    {
                        "id": doc.entry.id,
                        "work_id": doc.entry.work_id,
                        "author": doc.entry.effective_author,
                        "split": doc.entry.split,
                        "source_hash": doc.source_hash,
                        "normalized_hash": doc.normalized_hash,
                        "warnings": doc.warnings,
                    }
                    for doc in loaded
                ],
            },
            None,
        )
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        _fail(error)


@app.command("extract")
def extract(
    manifest: Path,
    author: str = typer.Option(..., "--author", "-a"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    corrections: Optional[Path] = typer.Option(None, help="JSON list of writer corrections."),
    annotations: Optional[Path] = typer.Option(None, help="JSON list of candidate traits with exact source evidence."),
) -> None:
    """Build a profile from this author's included training pieces only."""
    from colophon.voice.manifest import load_corpus
    from colophon.voice.profile import build_voice_profile

    try:
        edits = (
            [TraitCorrection.model_validate(item) for item in json.loads(corrections.read_text())]
            if corrections
            else None
        )
        traits = (
            [VoiceTrait.model_validate(item) for item in json.loads(annotations.read_text())] if annotations else None
        )
        profile = build_voice_profile(load_corpus(manifest), author, corrections=edits, annotations=traits)
        _write(profile.model_dump_json(indent=2), output)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        _fail(error)


@app.command("inspect")
def inspect_profile(path: Path, evidence: bool = typer.Option(False, "--evidence")) -> None:
    """Inspect a saved profile and optionally print its exact supporting passages."""
    try:
        profile = VoiceProfile.model_validate_json(path.read_text())
        if evidence:
            from colophon.voice.export import export_voice_markdown

            _write(export_voice_markdown(profile), None)
        else:
            _json(
                {
                    "profile_id": profile.profile_id,
                    "author": profile.author,
                    "language": profile.language,
                    "documents": len(profile.documents),
                    "traits": [
                        {
                            "id": trait.id,
                            "status": trait.status,
                            "statement": trait.statement,
                            "supporting_documents": sorted({span.document_id for span in trait.evidence}),
                        }
                        for trait in profile.traits
                    ],
                    "warnings": profile.warnings,
                    "limitations": profile.limitations,
                },
                None,
            )
    except (ValueError, OSError) as error:
        _fail(error)


@app.command("export")
def export_profile(
    path: Path,
    format: str = typer.Option("markdown", "--format", "-f"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    all_evidence: bool = typer.Option(
        False, "--all-evidence", help="Include every source/evidence record in Markdown."
    ),
) -> None:
    """Export a portable profile as Markdown or JSON."""
    try:
        profile = VoiceProfile.model_validate_json(path.read_text())
        if format == "json":
            _write(profile.model_dump_json(indent=2), output)
        elif format == "markdown":
            from colophon.voice.export import export_voice_markdown

            _write(
                export_voice_markdown(
                    profile,
                    evidence_limit=None if all_evidence else 3,
                    document_limit=None if all_evidence else 12,
                ),
                output,
            )
        else:
            raise ValueError("Export format must be markdown or json")
    except (ValueError, OSError) as error:
        _fail(error)


@app.command("prompt")
def prompt(
    path: Path,
    mode: str = typer.Option("hybrid", help="numeric, exemplars, qualitative or hybrid"),
    task: str = typer.Option("rewrite", help="rewrite (default) or generate new writing from supplied facts"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    receipt: bool = typer.Option(False, "--receipt", help="Require -o and report exact artifact readback as JSON."),
) -> None:
    """Export writing guidance for any provider; never sends text or calls a model."""
    import hashlib

    from colophon.voice.export import build_voice_prompt

    record: dict[str, object] = {
        "schema_version": "1",
        "status": "not_written",
        "target_path": str(output.absolute()) if output is not None else None,
        "artifact": None,
        "source_profile": None,
        "mode": mode,
        "task": task,
        "verification_scope": "prompt artifact only",
        "quality_dimensions": dict.fromkeys(
            ["voice", "naturalness", "factuality", "preservation", "copying", "human_or_literary_validation"],
            "unverified",
        ),
        "next_permitted_step": (
            "Use the saved prompt for external generation using supplied facts; "
            "exporting never called a writing model."
        ),
    }

    def fail(error: Exception, *, status: str, recovery: str) -> None:
        if receipt:
            record.update(status=status, error=str(error), recovery=recovery, next_permitted_step=recovery)
            _json(record, None)
            raise typer.Exit(1)
        _fail(error)

    try:
        if receipt and output is None:
            raise ValueError("--receipt requires --output/-o; choose a separate prompt artifact path")
        if output is not None and (
            output.resolve() == path.resolve() or (output.exists() and output.samefile(path))
        ):
            raise ValueError("Prompt output aliases the source profile; choose a different output path")
        source = path.read_bytes()
        profile = VoiceProfile.model_validate_json(source)
        record["source_profile"] = {
            "id": profile.profile_id,
            "path": str(path.absolute()),
            "sha256": hashlib.sha256(source).hexdigest(),
        }
        rendered = build_voice_prompt(profile, mode=mode, task=task)
    except (ValueError, OSError, RuntimeError) as error:
        fail(
            error,
            status="not_written",
            recovery="Check the source profile, mode/task and a separate -o path, then rerun; no write was attempted.",
        )
        return

    try:
        _write(rendered, output)
    except (ValueError, OSError) as error:
        fail(
            error,
            status="unknown",
            recovery="Inspect the target before retrying; artifact replacement could not be verified.",
        )
        return
    if receipt and output is not None:
        try:
            observed = output.read_bytes()
        except OSError as error:
            fail(
                error,
                status="unknown",
                recovery="Retain and inspect the saved artifact; restore read access and compare before any retry.",
            )
            return
        record["artifact"] = {"bytes": len(observed), "sha256": hashlib.sha256(observed).hexdigest()}
        expected = (rendered if rendered.endswith("\n") else rendered + "\n").encode("utf-8")
        if observed != expected:
            fail(
                ValueError("Saved prompt bytes do not match the rendered prompt"),
                status="unverified",
                recovery="Retain and inspect the artifact and source; compare their contents before any retry.",
            )
            return
        record["status"] = "written"
        _json(record, None)


@app.command("evaluate")
def evaluate(
    manifest: Path,
    split: str = typer.Option("test", help="Held-out dev or test split."),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """Compare four offline retrieval baselines on unseen documents, without generation."""
    from colophon.voice.evaluation import evaluate_corpus
    from colophon.voice.manifest import load_corpus

    try:
        _json(evaluate_corpus(load_corpus(manifest), split=split), output)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        _fail(error)


@app.command("evaluate-writing")
def evaluate_writing_command(
    cases: Path,
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """Score supplied outputs from JSON or JSONL; retain ungraded and error states."""
    from colophon.voice.evaluation import evaluate_writing

    try:
        content = cases.read_text()
        records = (
            [json.loads(line) for line in content.splitlines() if line.strip()]
            if cases.suffix == ".jsonl"
            else json.loads(content)
        )
        if not isinstance(records, list):
            raise ValueError("Writing cases must be a JSON array or JSONL records")
        _json(evaluate_writing(records), output)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        _fail(error)


@app.command("evaluate-profile")
def evaluate_profile_command(
    profile: Path,
    manifest: Path,
    split: str = typer.Option("test", help="Held-out dev or test split."),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """Check one author's profile against unseen pieces; not authorship classification."""
    from colophon.voice.manifest import load_corpus
    from colophon.voice.profile_validation import evaluate_profile

    try:
        saved = VoiceProfile.model_validate_json(profile.read_text())
        _json(evaluate_profile(saved, load_corpus(manifest), split=split), output)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        _fail(error)


def doctor() -> None:
    """Report readiness without installing packages or reading API keys."""
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ["spacy", "en_core_web_sm", "textstat", "numpy", "ebooklib", "bs4", "fitz", "pytest"]
    }
    _json(
        {
            "packages": packages,
            "ready_for_english_voice": all(packages[name] for name in ["spacy", "en_core_web_sm", "textstat", "numpy"]),
            "note": "Package presence is not parser quality or voice-fidelity validation.",
        },
        None,
    )

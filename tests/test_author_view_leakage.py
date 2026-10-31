"""Synthetic corpus-boundary checks for the prose actually used by analysis."""

import json
from pathlib import Path
from typing import Literal

import pytest
from typer.testing import CliRunner

from colophon.cli import app
from colophon.models.document import Document
from colophon.voice.manifest import load_corpus, validate_loaded


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def write_manifest(root: Path, entries: list[dict]) -> Path:
    path = root / "corpus.json"
    path.write_text(json.dumps({"name": "Synthetic boundary corpus", "documents": entries}))
    return path


def write_source(
    root: Path,
    name: str,
    body: str,
    *,
    split: str = "train",
    kind: Literal["heading", "furniture", "quotation", "inline", "span", "editor"] = "heading",
    padding: int = 600,
    quotation_policy: str = "exclude_all",
) -> dict:
    excluded = words(name, padding)
    text = f'{excluded}\n\n{body}' if kind != "inline" else f'"{excluded}"\n\n{body}'
    document = Document.from_text(text, f"{name}.txt", metadata={"fixture": "synthetic"})
    if kind in {"heading", "furniture", "quotation", "editor"}:
        update = {"attribution": "editor"} if kind == "editor" else {"kind": kind}
        document = document.model_copy(
            update={"blocks": [document.blocks[0].model_copy(update=update), *document.blocks[1:]]}
        )
    source = root / f"{name}.colophon.json"
    source.write_text(document.model_dump_json())
    entry: dict[str, object] = {
        "id": name,
        "path": source.name,
        "author": "Synthetic",
        "work_id": name,
        "split": split,
        "quotation_policy": quotation_policy,
    }
    if kind == "span":
        entry["excluded_spans"] = [{"start": 0, "end": len(excluded), "reason": "Synthetic excluded header"}]
    return entry


def test_public_cli_rejects_author_body_hidden_by_distinct_large_headings(tmp_path: Path):
    body = words("body", 100)
    entries = []
    for name, split, count in [("training", "train", 600), ("heldout", "test", 900)]:
        source = tmp_path / f"{name}.md"
        source.write_text(f"# {words(name, count)}\n\n{body}")
        entries.append({"id": name, "path": source.name, "author": "Synthetic", "work_id": name, "split": split})
    manifest = write_manifest(tmp_path, entries)

    result = CliRunner().invoke(app, ["corpus", "validate", str(manifest)])

    assert result.exit_code == 1, result.output
    assert "Duplicate author text" in result.output
    assert "training" in result.output and "heldout" in result.output


@pytest.mark.parametrize("split", ["train", "dev", "test"])
@pytest.mark.parametrize("body", ["One short sentence.", words("body", 100)])
def test_exact_author_duplicates_ignore_padding_case_and_whitespace(tmp_path: Path, split: str, body: str):
    manifest = write_manifest(tmp_path, [
        write_source(tmp_path, "first", body, padding=200),
        write_source(tmp_path, "second", body.upper().replace(" ", " \t "), split=split, padding=900),
    ])
    with pytest.raises(ValueError, match="Duplicate author text"):
        load_corpus(manifest)


@pytest.mark.parametrize("kind", ["heading", "furniture", "quotation", "inline", "span", "editor"])
def test_each_exclusion_route_cannot_hide_duplicate_author_prose(tmp_path: Path, kind):
    manifest = write_manifest(tmp_path, [
        write_source(tmp_path, "first", words("body", 100), kind=kind),
        write_source(tmp_path, "second", words("body", 100), kind=kind, split="test", padding=900),
    ])
    with pytest.raises(ValueError, match="Duplicate author text"):
        load_corpus(manifest)


def test_explicit_quotation_blocks_cannot_hide_author_duplicates(tmp_path: Path):
    manifest = write_manifest(tmp_path, [
        write_source(
            tmp_path, name, words("body", 100), kind="quotation", split=split, quotation_policy="explicit_only"
        )
        for name, split in [("first", "train"), ("second", "test")]
    ])
    with pytest.raises(ValueError, match="Duplicate author text"):
        load_corpus(manifest)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("relationship", ["excerpt", "near"])
def test_cross_split_author_containment_and_near_duplicates(tmp_path: Path, reverse: bool, relationship: str):
    body = words("body", 100)
    other = f"{words('opening', 50)} {body} {words('ending', 50)}"
    if relationship == "near":
        other = " ".join(body.split()[:90]) + " " + words("revised", 10)
    first, second = (other, body) if reverse else (body, other)
    manifest = write_manifest(tmp_path, [
        write_source(tmp_path, "first", first),
        write_source(tmp_path, "second", second, split="test"),
    ])
    with pytest.raises(ValueError, match="Near-duplicate author text crosses splits"):
        load_corpus(manifest)


def test_same_split_author_excerpt_is_allowed(tmp_path: Path):
    body = words("body", 100)
    manifest = write_manifest(tmp_path, [
        write_source(tmp_path, "first", body),
        write_source(tmp_path, "second", f"{words('opening', 50)} {body}"),
    ])
    assert len(load_corpus(manifest)) == 2


def test_distinct_author_bodies_with_shared_quotes_keep_provenance(tmp_path: Path):
    quote = f'"{words("quoted", 100)}"'
    entries = [
        write_source(tmp_path, name, f"{quote}\n\n{words(name + 'body', 100)}", split=split)
        for name, split in [("first", "train"), ("second", "test")]
    ]
    loaded = load_corpus(write_manifest(tmp_path, entries))
    before = [item.document.model_dump() for item in loaded], [item.author_document.model_dump() for item in loaded]

    validate_loaded(loaded)

    assert before == (
        [item.document.model_dump() for item in loaded],
        [item.author_document.model_dump() for item in loaded],
    )
    for item in loaded:
        source = Document.model_validate_json((tmp_path / item.entry.path).read_text())
        assert item.document == source
        assert item.entry.work_id == item.entry.id
        assert len(item.author_document.text) == len(source.text)
        assert item.author_document.metadata["source_document_id"] == source.id
        assert item.author_document.metadata["fixture"] == "synthetic"
        assert item.author_document.metadata["source_hash"] == item.normalized_hash
        assert "quoted0" not in item.author_document.text
        body_block = item.author_document.blocks[-1]
        assert body_block == source.blocks[-1]
        assert item.author_document.text[body_block.char_offset_start:body_block.char_offset_end] == body_block.text
        assert item.excluded_word_count == 700


def test_full_source_duplicate_still_fails_with_distinct_author_views(tmp_path: Path):
    text = f"{words('first', 100)}\n\n{words('second', 100)}"
    entries = []
    ranges = [("first", 0, len(words("first", 100))), ("second", len(words("first", 100)) + 2, len(text))]
    for name, start, end in ranges:
        source = tmp_path / f"{name}.txt"
        source.write_text(text)
        entries.append({
            "id": name, "path": source.name, "author": "Synthetic", "work_id": name,
            "excluded_spans": [{"start": start, "end": end, "reason": "Synthetic excluded paragraph"}],
        })
    with pytest.raises(ValueError, match="Duplicate text"):
        load_corpus(write_manifest(tmp_path, entries))


def test_full_source_overlap_still_fails_with_distinct_author_views(tmp_path: Path):
    entries = []
    for name, split in [("first", "train"), ("second", "test")]:
        source = tmp_path / f"{name}.txt"
        source.write_text(f'"{words("quoted", 600)}"\n\n{words(name, 40)}')
        entries.append({"id": name, "path": source.name, "author": "Synthetic", "work_id": name, "split": split})
    with pytest.raises(ValueError, match="Near-duplicate text crosses splits"):
        load_corpus(write_manifest(tmp_path, entries))

"""Public CLI receipt checks: artifact evidence must not become a writing-quality claim."""

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from colophon.cli import app
from colophon.voice.export import build_voice_prompt
from colophon.voice.models import VoiceProfile

runner = CliRunner()


@pytest.fixture
def saved_profile(tmp_path):
    profile = VoiceProfile(
        profile_id="receipt-test",
        author="Synthetic writer",
        language="en",
        corpus_hash="corpus",
        config_hash="config",
        documents=[],
    )
    source = tmp_path / "source.json"
    # Preserve byte identity independently of parsed/serialized JSON identity.
    source.write_bytes((profile.model_dump_json(indent=2) + "\r\n").encode("utf-8"))
    return source, profile


@pytest.mark.parametrize("task", ["rewrite", "generate"])
def test_receipt_reads_saved_artifact_and_source_exactly_once(saved_profile, tmp_path, monkeypatch, task):
    source, profile = saved_profile
    target = tmp_path / "prompt.txt"
    source_bytes = source.read_bytes()
    original_read = Path.read_bytes
    reads = []

    def read_bytes(path):
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    result = runner.invoke(app, [
        "voice", "prompt", str(source), "--task", task, "--mode", "exemplars", "-o", str(target), "--receipt",
    ])
    assert result.exit_code == 0, result.output
    receipt = json.loads(result.stdout)
    assert reads == [source, target]
    observed = original_read(target)
    assert observed == (build_voice_prompt(profile, "exemplars", task=task) + "\n").encode()
    assert receipt["schema_version"] == "1"
    assert receipt["status"] == "written"
    assert receipt["target_path"] == str(target.absolute())
    assert receipt["artifact"] == {"bytes": len(observed), "sha256": hashlib.sha256(observed).hexdigest()}
    assert receipt["source_profile"] == {
        "id": profile.profile_id, "path": str(source.absolute()), "sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    assert receipt["mode"] == "exemplars"
    assert receipt["task"] == task
    assert receipt["verification_scope"] == "prompt artifact only"
    assert set(receipt["quality_dimensions"].values()) == {"unverified"}
    assert "human_or_literary_validation" in receipt["quality_dimensions"]
    assert "supplied facts" in receipt["next_permitted_step"]
    assert "never called a writing model" in receipt["next_permitted_step"]


def test_default_stdout_and_silent_file_output_remain_exact(saved_profile, tmp_path):
    source, profile = saved_profile
    expected = build_voice_prompt(profile) + "\n"
    result = runner.invoke(app, ["voice", "prompt", str(source)])
    assert result.exit_code == 0, result.output
    assert result.stdout == expected
    target = tmp_path / "prompt.txt"
    result = runner.invoke(app, ["voice", "prompt", str(source), "-o", str(target)])
    assert result.exit_code == 0, result.output
    assert result.output == ""
    assert target.read_text() == expected


@pytest.mark.parametrize("options", [["--mode", "invalid"], ["--task", "invalid"]])
def test_invalid_option_returns_no_write_receipt_and_retains_target(saved_profile, tmp_path, options):
    source, _ = saved_profile
    target = tmp_path / "prompt.txt"
    target.write_bytes(b"retained")
    result = runner.invoke(app, ["voice", "prompt", str(source), "-o", str(target), "--receipt", *options])
    assert result.exit_code == 1
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "not_written"
    assert receipt["artifact"] is None
    assert "no write was attempted" in receipt["recovery"]
    assert target.read_bytes() == b"retained"


def test_receipt_without_output_refuses_before_reading_source(saved_profile, monkeypatch):
    source, _ = saved_profile

    def unexpected_read(path):
        pytest.fail("Missing output must be rejected before reading the source")

    monkeypatch.setattr(Path, "read_bytes", unexpected_read)
    result = runner.invoke(app, ["voice", "prompt", str(source), "--receipt"])
    assert result.exit_code == 1
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "not_written"
    assert receipt["target_path"] is None
    assert "requires --output/-o" in receipt["error"]


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
@pytest.mark.parametrize("receipt_mode", [False, True])
def test_source_alias_is_refused_and_preserved(saved_profile, tmp_path, alias, receipt_mode):
    source, _ = saved_profile
    before = source.read_bytes()
    target = source if alias == "same" else tmp_path / "alias.json"
    if alias == "symlink":
        target.symlink_to(source)
    elif alias == "hardlink":
        target.hardlink_to(source)
    options = ["--receipt"] if receipt_mode else []
    result = runner.invoke(app, ["voice", "prompt", str(source), "-o", str(target), *options])
    assert result.exit_code == 1
    assert "aliases the source profile" in result.output
    if receipt_mode:
        assert json.loads(result.stdout)["status"] == "not_written"
    assert source.read_bytes() == before
    assert target.read_bytes() == before
    if alias == "symlink":
        assert target.is_symlink()


@pytest.mark.parametrize("unreadable", [False, True])
def test_failed_readback_never_claims_written_or_retries(saved_profile, tmp_path, monkeypatch, unreadable):
    source, profile = saved_profile
    target = tmp_path / "prompt.txt"
    original_read = Path.read_bytes
    readbacks = []

    def read_bytes(path):
        if path == target:
            readbacks.append(path)
            assert original_read(path) == (build_voice_prompt(profile) + "\n").encode()
            if unreadable:
                raise PermissionError("simulated unreadable readback")
            return b"unexpected observed content"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    result = runner.invoke(app, ["voice", "prompt", str(source), "-o", str(target), "--receipt"])
    assert result.exit_code == 1
    receipt = json.loads(result.stdout)
    assert receipt["status"] == ("unknown" if unreadable else "unverified")
    assert readbacks == [target]
    assert "Retain and inspect" in receipt["recovery"]
    assert target.exists()
    assert original_read(target) == (build_voice_prompt(profile) + "\n").encode()
    assert set(receipt["quality_dimensions"].values()) == {"unverified"}
    if unreadable:
        assert receipt["artifact"] is None
    else:
        observed = b"unexpected observed content"
        assert receipt["artifact"] == {"bytes": len(observed), "sha256": hashlib.sha256(observed).hexdigest()}


@pytest.mark.parametrize("invalid_source", [False, True])
def test_source_read_or_validation_failure_has_no_write_receipt(tmp_path, invalid_source):
    source = tmp_path / "source.json"
    if invalid_source:
        source.write_bytes(b"invalid JSON")
    target = tmp_path / "prompt.txt"
    target.write_bytes(b"retained")
    result = runner.invoke(app, ["voice", "prompt", str(source), "-o", str(target), "--receipt"])
    assert result.exit_code == 1
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "not_written"
    assert receipt["source_profile"] is None
    assert receipt["recovery"]
    assert target.read_bytes() == b"retained"

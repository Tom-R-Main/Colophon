"""Public agent route: discover, resolve, export and verify a private profile."""
import hashlib
import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from colophon.cli import app

ROOT = Path(__file__).parents[1]
RUNNER = CliRunner()


def test_native_discovery_resolve_and_prompt_receipt(tmp_path):
    sources = tmp_path / "sources"
    shutil.copytree(ROOT / "examples/voice-corpus", sources)
    profile = tmp_path / "writer.voice.json"
    extracted = RUNNER.invoke(app, [
        "voice", "extract", str(sources / "corpus.json"), "--author", "Rowan (synthetic)", "-o", str(profile),
    ])
    assert extracted.exit_code == 0, extracted.output
    registry = tmp_path / "profiles.json"
    registry.write_text(json.dumps({"schema_version": 1, "profiles": {"my-voice": {"path": str(profile)}}}))

    listed = RUNNER.invoke(app, ["voice", "profiles", "list", "--registry", str(registry)])
    assert listed.exit_code == 0, listed.output
    discovery = json.loads(listed.output)
    assert discovery["status"] == "configured"
    assert discovery["profiles"] == [{"name": "my-voice", "kind": "VoiceProfile"}]
    assert "exemplars" not in listed.output and "supporting_documents" not in listed.output

    selected = RUNNER.invoke(app, ["voice", "profiles", "resolve", "my-voice", "--registry", str(registry)])
    assert selected.exit_code == 0, selected.output
    resolved = json.loads(selected.output)
    source = json.loads(profile.read_text())
    assert resolved["status"] == "resolved" and resolved["kind"] == "VoiceProfile"
    assert resolved["profile_path"] == str(profile.resolve())
    assert resolved["profile_sha256"] == hashlib.sha256(profile.read_bytes()).hexdigest()
    assert resolved["profile_id"] == source["profile_id"]
    assert resolved["documents"] == 2 and resolved["language"] == "en"
    assert "exemplars" not in resolved and "traits" not in resolved and "author" not in resolved

    prompt = tmp_path / "readme.prompt.txt"
    exported = RUNNER.invoke(app, [
        "voice", "prompt", resolved["profile_path"], "--task", "rewrite", "--mode", "exemplars",
        "-o", str(prompt), "--receipt",
    ])
    assert exported.exit_code == 0, exported.output
    receipt = json.loads(exported.output)
    assert receipt["status"] == "written"
    assert receipt["source_profile"]["id"] == resolved["profile_id"]
    assert receipt["source_profile"]["sha256"] == resolved["profile_sha256"]
    assert receipt["artifact"]["sha256"] == hashlib.sha256(prompt.read_bytes()).hexdigest()
    assert set(receipt["quality_dimensions"].values()) == {"unverified"}


def test_registry_cli_missing_and_unknown_have_machine_readable_recovery(tmp_path):
    missing = tmp_path / "absent.json"
    listed = RUNNER.invoke(app, ["voice", "profiles", "list", "--registry", str(missing)])
    assert listed.exit_code == 0
    assert json.loads(listed.output)["status"] == "not_configured"
    resolved = RUNNER.invoke(app, ["voice", "profiles", "resolve", "my-voice", "--registry", str(missing)])
    assert resolved.exit_code == 1
    error = json.loads(resolved.output)
    assert error["status"] == "error" and error["error"]["code"] == "registry_not_found"
    assert error["recovery"]
    registry = tmp_path / "profiles.json"
    registry.write_text('{"schema_version":1,"profiles":{}}')
    resolved = RUNNER.invoke(app, ["voice", "profiles", "resolve", "unknown", "--registry", str(registry)])
    assert resolved.exit_code == 1
    assert json.loads(resolved.output)["error"]["code"] == "unknown_profile"


def test_registry_cli_wrong_profile_kind_does_not_echo_private_contents(tmp_path):
    profile = tmp_path / "legacy.analysis.json"
    profile.write_text(json.dumps({"document_id": "legacy", "document_title": "PRIVATE_SENTINEL_NO_ECHO"}))
    registry = tmp_path / "profiles.json"
    registry.write_text(json.dumps({"schema_version": 1, "profiles": {"my-voice": {"path": str(profile)}}}))
    result = RUNNER.invoke(app, ["voice", "profiles", "resolve", "my-voice", "--registry", str(registry)])
    assert result.exit_code == 1
    assert json.loads(result.output)["error"]["code"] == "invalid_profile"
    assert "PRIVATE_SENTINEL_NO_ECHO" not in result.output

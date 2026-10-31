"""Synthetic local registry cases; never inspect the user's real configuration."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from colophon.voice import registry as api

SENTINEL = "PRIVATE-DISTINCTIVE-SENTINEL"


def git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init")
    git(root, "config", "core.excludesFile", "")
    return root


@pytest.fixture
def profile_data():
    return {
        "schema_version": "1",
        "extractor_version": "1",
        "profile_id": "synthetic-profile",
        "author": SENTINEL,
        "language": "en",
        "corpus_hash": "synthetic-corpus",
        "config_hash": "synthetic-config",
        "documents": [{
            "entry": {"id": "one", "path": f"/{SENTINEL}.txt", "author": SENTINEL, "work_id": "one"},
            "source_path": f"/{SENTINEL}.txt",
            "source_hash": "source-hash",
            "normalized_hash": "normalized-hash",
            "measurement": {"document_id": "one", "document_title": SENTINEL},
            "author_word_count": 10,
            "excluded_word_count": 0,
        }],
        "traits": [{"id": "trait", "category": "stance", "statement": SENTINEL}],
        "exemplars": [{
            "document_id": "one", "source_hash": "source-hash", "block_id": "one",
            "start": 0, "end": len(SENTINEL), "text": SENTINEL,
        }],
    }


def configure(path: Path, profiles: dict[str, Path]) -> Path:
    path.write_text(json.dumps({
        "schema_version": 1,
        "profiles": {name: {"path": str(source)} for name, source in profiles.items()},
    }))
    return path


def assert_code(code, function, *arguments):
    with pytest.raises(api.RegistryError) as caught:
        function(*arguments)
    error = caught.value
    assert error.code == code
    assert error.message and error.recovery
    assert SENTINEL not in str(error)
    assert SENTINEL not in error.recovery


def test_path_precedence_and_missing_state(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.json"
    environment = tmp_path / "environment.json"
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("COLOPHON_PROFILES_FILE", str(environment))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    result = api.list_profiles(explicit)
    assert result == {
        "schema_version": 1, "status": "not_configured", "registry_path": str(explicit),
        "profiles": [], "recovery": result["recovery"],
    }
    assert api.list_profiles()["registry_path"] == str(environment)
    monkeypatch.delenv("COLOPHON_PROFILES_FILE")
    assert api.list_profiles()["registry_path"] == str(xdg / "colophon" / "profiles.json")
    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert api.list_profiles()["registry_path"] == str(tmp_path / ".config" / "colophon" / "profiles.json")
    assert_code("registry_not_found", api.resolve_profile, "voice", explicit)


def test_tilde_paths_expand_without_cwd_or_environment_interpolation(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    configure(tmp_path / "profiles.json", {"my-voice": Path("~/writer.json")})
    assert api.list_profiles(Path("~/profiles.json"))["registry_path"] == str(tmp_path / "profiles.json")
    assert_code("profile_not_found", api.resolve_profile, "my-voice", Path("~/profiles.json"))
    for value in ("relative.json", "$HOME/profiles.json", "/tmp/*.json", "/tmp/[x].json"):
        assert_code("invalid_registry", api.list_profiles, Path(value))
    monkeypatch.setenv("COLOPHON_PROFILES_FILE", "relative.json")
    assert_code("invalid_registry", api.list_profiles)
    monkeypatch.delenv("COLOPHON_PROFILES_FILE")
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative")
    assert_code("invalid_registry", api.list_profiles)


@pytest.mark.parametrize("contents", [
    "{", "[]", "null", '"text"',
    '{"schema_version":true,"profiles":{}}',
    '{"schema_version":"1","profiles":{}}',
    '{"schema_version":2,"profiles":{}}',
    '{"schema_version":1,"profiles":{},"other":1}',
    '{"schema_version":1,"schema_version":1,"profiles":{}}',
    '{"schema_version":1,"profiles":{"voice":{"path":"/tmp/a","path":"/tmp/b"}}}',
    '{"schema_version":1,"profiles":{"voice":{"path":"/tmp/a"},"voice":{"path":"/tmp/b"}}}',
    '{"schema_version":1,"profiles":{"voice":{"path":"/tmp/a","kind":"StyleProfile"}}}',
    '{"schema_version":1,"profiles":{"voice":{"path":"relative"}}}',
    '{"schema_version":1,"profiles":{"voice":{"path":null}}}',
    '{"schema_version":1,"profiles":{"voice":[]}}',
    '{"schema_version":1,"profiles":[]}',
    '{"schema_version":1,"profiles":{},"secret":NaN}',
])
def test_invalid_registry_is_safe(tmp_path, contents):
    path = tmp_path / f"{SENTINEL}.json"
    path.write_text(contents)
    assert_code("invalid_registry", api.list_profiles, path)


@pytest.mark.parametrize("alias", ["", "Upper", "two words", "-voice", "_voice", "a.b", "é", "x" * 65])
def test_invalid_aliases_reject_entire_registry(tmp_path, alias):
    path = configure(tmp_path / "registry.json", {alias: tmp_path / "voice.json"})
    assert_code("invalid_registry", api.list_profiles, path)


def test_list_reads_no_profile_bytes_and_exposes_only_aliases(tmp_path, monkeypatch):
    target = tmp_path / f"{SENTINEL}.json"
    target.write_text(SENTINEL)
    path = configure(tmp_path / "registry.json", {"z": target, "a" * 64: target, "0_voice-x": target})
    actual = api._read_private
    calls = []

    def read(source, kind):
        calls.append(kind)
        return actual(source, kind)

    monkeypatch.setattr(api, "_read_private", read)
    result = api.list_profiles(path)
    assert calls == ["registry"]
    assert result["profiles"] == [
        {"name": name, "kind": "VoiceProfile"} for name in ("0_voice-x", "a" * 64, "z")
    ]
    assert result["validation_scope"] == "registry entries only"
    assert SENTINEL not in json.dumps(result)
    target.unlink()
    assert api.list_profiles(path) == result


def test_resolve_reads_once_hashes_actual_bytes_and_returns_only_metadata(tmp_path, profile_data, monkeypatch):
    source = tmp_path / "writer.json"
    raw = (json.dumps(profile_data, indent=3) + "\n").encode()
    source.write_bytes(raw)
    path = configure(tmp_path / "registry.json", {"my-voice": source})
    actual = api._read_private
    calls = []

    def read(selected, kind):
        calls.append(kind)
        return actual(selected, kind)

    monkeypatch.setattr(api, "_read_private", read)
    result = api.resolve_profile("my-voice", path)
    assert calls == ["registry", "profile"]
    assert result == {
        "schema_version": 1, "status": "resolved", "name": "my-voice", "kind": "VoiceProfile",
        "registry_path": str(path), "profile_path": str(source.resolve()),
        "profile_id": "synthetic-profile", "profile_sha256": hashlib.sha256(raw).hexdigest(),
        "language": "en", "documents": 1, "validation_scope": "profile schema and local file identity",
    }
    assert SENTINEL not in json.dumps(result)
    assert_code("unknown_profile", api.resolve_profile, SENTINEL, path)


@pytest.mark.parametrize("mutation", ["schema", "extractor", "missing_version", "document", "extra", "evidence"])
def test_invalid_full_profile_has_no_content_diagnostics(tmp_path, profile_data, mutation):
    if mutation == "schema":
        profile_data["schema_version"] = "2"
    elif mutation == "extractor":
        profile_data["extractor_version"] = "2"
    elif mutation == "missing_version":
        del profile_data["schema_version"]
    elif mutation == "document":
        profile_data["documents"][0]["author_word_count"] = SENTINEL
    elif mutation == "extra":
        profile_data["secret"] = SENTINEL
    else:
        profile_data["exemplars"][0]["end"] = 1
    source = tmp_path / "writer.json"
    source.write_text(json.dumps(profile_data))
    path = configure(tmp_path / "registry.json", {"voice": source})
    assert_code("invalid_profile", api.resolve_profile, "voice", path)


@pytest.mark.parametrize("contents", [
    "{", "[]", "null", '{"document_id":"old","document_title":"StyleProfile"}',
    '{"schema_version":"1","extractor_version":"1","profile_id":"a","profile_id":"b"}',
])
def test_legacy_malformed_and_duplicate_profile_rejected_without_fallback(tmp_path, contents):
    source = tmp_path / "writer.json"
    source.write_text(contents)
    path = configure(tmp_path / "registry.json", {"voice": source})
    assert_code("invalid_profile", api.resolve_profile, "voice", path)


@pytest.mark.parametrize("kind", ["registry", "profile"])
@pytest.mark.parametrize("state", ["tracked", "nonignored", "ignored"])
def test_git_private_policy(repo, tmp_path, profile_data, kind, state):
    source = (repo if kind == "profile" else tmp_path) / "writer.json"
    path = (repo if kind == "registry" else tmp_path) / "registry.json"
    source.write_text(json.dumps(profile_data))
    configure(path, {"voice": source})
    selected = path if kind == "registry" else source
    if state in {"tracked", "ignored"}:
        (repo / ".gitignore").write_text(selected.name + "\n")
    if state == "tracked":
        git(repo, "add", "--force", selected.name)
    function, arguments = (api.list_profiles, [path]) if kind == "registry" else (api.resolve_profile, ["voice", path])
    if state == "ignored":
        assert function(*arguments)["status"] in {"configured", "resolved"}
    else:
        assert_code(f"{kind}_not_private", function, *arguments)


def test_listing_does_not_validate_profile_privacy(repo, tmp_path):
    source = repo / "writer.json"
    source.write_text(SENTINEL)
    git(repo, "add", "writer.json")
    path = configure(tmp_path / "registry.json", {"voice": source})
    assert api.list_profiles(path)["status"] == "configured"
    assert_code("profile_not_private", api.resolve_profile, "voice", path)


def test_missing_paths_in_git_checkout_report_missing(repo, tmp_path):
    assert api.list_profiles(repo / "missing-registry.json")["status"] == "not_configured"
    path = configure(tmp_path / "registry.json", {"voice": repo / "missing-profile.json"})
    assert_code("profile_not_found", api.resolve_profile, "voice", path)


def test_symlink_targets_and_lexical_locations_both_obey_git_policy(repo, tmp_path, profile_data):
    source = repo / "writer.json"
    source.write_text(json.dumps(profile_data))
    outside = tmp_path / "link.json"
    outside.symlink_to(source)
    path = configure(tmp_path / "registry.json", {"voice": outside})
    assert_code("profile_not_private", api.resolve_profile, "voice", path)
    (repo / ".gitignore").write_text("writer.json\n")
    assert api.resolve_profile("voice", path)["profile_path"] == str(source)
    git(repo, "add", "--force", "writer.json")
    assert_code("profile_not_private", api.resolve_profile, "voice", path)
    source = tmp_path / "private.json"
    source.write_text(json.dumps(profile_data))
    inside = repo / "inside.json"
    inside.symlink_to(source)
    configure(path, {"voice": inside})
    assert_code("profile_not_private", api.resolve_profile, "voice", path)
    (repo / ".gitignore").write_text("inside.json\n")
    assert api.resolve_profile("voice", path)["profile_path"] == str(source)


def test_parent_symlink_cannot_bypass_tracked_status(repo, tmp_path, profile_data):
    private = tmp_path / "private"
    private.mkdir()
    source = private / "writer.json"
    source.write_text(json.dumps(profile_data))
    link = repo / "link"
    link.symlink_to(private, target_is_directory=True)
    (repo / ".gitignore").write_text("*\n")
    git(repo, "add", "--force", "link")
    path = configure(tmp_path / "registry.json", {"voice": link / "writer.json"})
    assert_code("profile_not_private", api.resolve_profile, "voice", path)


def test_git_metadata_and_symlink_detours_are_forbidden(repo, tmp_path):
    path = configure(tmp_path / "registry.json", {"voice": repo / ".git" / "config"})
    assert_code("profile_not_private", api.resolve_profile, "voice", path)
    link = tmp_path / "metadata"
    link.symlink_to(repo / ".git", target_is_directory=True)
    configure(path, {"voice": link / "config"})
    assert_code("profile_not_private", api.resolve_profile, "voice", path)
    assert_code("registry_not_private", api.list_profiles, repo / ".git" / "config")


def test_unverifiable_git_metadata_fails_closed(tmp_path):
    fake = tmp_path / "broken-checkout"
    fake.mkdir()
    (fake / ".git").write_text("not valid git metadata")
    path = configure(fake / "registry.json", {})
    assert_code("privacy_check_failed", api.list_profiles, path)


def test_git_timeout_is_safe_and_closed(repo, monkeypatch):
    path = configure(repo / "registry.json", {})

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("git", 5, output=SENTINEL)

    monkeypatch.setattr(api.subprocess, "run", timeout)
    assert_code("privacy_check_failed", api.list_profiles, path)


def test_directory_and_missing_profile_are_typed_errors(tmp_path):
    path = configure(tmp_path / "registry.json", {"voice": tmp_path / "missing.json"})
    assert_code("profile_not_found", api.resolve_profile, "voice", path)
    configure(path, {"voice": tmp_path})
    assert_code("invalid_profile", api.resolve_profile, "voice", path)
    assert_code("invalid_registry", api.list_profiles, tmp_path)


@pytest.mark.parametrize("flag", ["O_NOFOLLOW", "O_NONBLOCK"])
def test_missing_safe_open_capability_is_typed_and_closed(tmp_path, monkeypatch, flag):
    path = configure(tmp_path / "registry.json", {})
    monkeypatch.delattr(api.os, flag)
    assert_code("privacy_check_failed", api.list_profiles, path)

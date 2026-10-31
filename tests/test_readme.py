"""The README's first run must produce the promised artifacts, not just parse."""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
from pathlib import Path
from urllib.parse import unquote, urlsplit

from typer.testing import CliRunner

from colophon.cli import app
from colophon.voice.models import VoiceProfile

ROOT = Path(__file__).parents[1]


def quickstart_commands(text: str) -> list[list[str]]:
    match = re.search(r"<!-- test:quickstart -->\s*```bash\n(.*?)```\s*<!-- /test:quickstart -->", text, re.S)
    if match is None:
        raise ValueError("README needs one marked executable quick start")
    commands = []
    for line in match[1].strip().splitlines():
        argv = shlex.split(line)
        if not argv or argv[0] != "colophon" or any(word in argv for word in (";", "|", ">", "&&")):
            raise ValueError("Quick start must contain plain Colophon commands")
        commands.append(argv[1:])
    if not commands:
        raise ValueError("Quick start cannot be empty")
    return commands


def test_readme_quickstart_has_real_artifact_postconditions(tmp_path, monkeypatch):
    fixture_target = tmp_path / "examples" / "voice-corpus"
    shutil.copytree(ROOT / "examples" / "voice-corpus", fixture_target)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    outputs = []
    for argv in quickstart_commands((ROOT / "README.md").read_text()):
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, f"Documented command failed: {argv}\n{result.output}"
        outputs.append(result.output)
    profile_path = tmp_path / "outputs/demo/writer.voice.json"
    profile = VoiceProfile.model_validate_json(profile_path.read_text())
    assert profile.author == "Rowan (synthetic)"
    assert {d.entry.id for d in profile.documents} == {"rowan-harbor", "rowan-library"}
    assert all(span.document_id != "rowan-garden" for span in profile.exemplars)
    report = (tmp_path / "outputs/demo/writer.voice.md").read_text()
    assert profile.profile_id in report and "characters" in report
    prompt = tmp_path / "outputs/demo/writer.prompt.txt"
    receipt = next(json.loads(text) for text in outputs if '"verification_scope"' in text)
    assert receipt["status"] == "written"
    assert receipt["target_path"] == str(prompt)
    assert receipt["artifact"]["sha256"] == hashlib.sha256(prompt.read_bytes()).hexdigest()
    assert receipt["source_profile"]["id"] == profile.profile_id
    assert receipt["mode"] == "exemplars" and receipt["task"] == "generate"
    assert set(receipt["quality_dimensions"].values()) == {"unverified"}


def test_quickstart_check_rejects_missing_block_or_shell_route():
    import pytest

    invalid_examples = (
        "No runnable example",
        "<!-- test:quickstart -->\n```bash\nsh install.sh\n```\n<!-- /test:quickstart -->",
    )
    for invalid in invalid_examples:
        with pytest.raises(ValueError):
            quickstart_commands(invalid)


def test_public_documentation_local_links_resolve():
    documents = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *sorted((ROOT / "docs").glob("*.md"))]
    for document in documents:
        for raw in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", document.read_text()):
            target = urlsplit(raw)
            if target.scheme or target.netloc:
                continue
            path = (document.parent / unquote(target.path)).resolve() if target.path else document
            assert path.is_relative_to(ROOT), f"Public link leaves the project: {document.name}"
            assert path.is_file(), f"Broken documentation link: {document.name} -> {raw}"
            if target.fragment:
                headings = re.findall(r"^#{1,6}\s+(.+)$", path.read_text(), re.M)
                anchors = {re.sub(r"[^\w-]", "", heading.lower().replace(" ", "-")) for heading in headings}
                assert unquote(target.fragment) in anchors, f"Missing heading: {document.name} -> {raw}"

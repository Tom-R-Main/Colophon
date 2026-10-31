"""Publication boundary tests use only invented content and temporary Git repos."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_publication.py"
SPEC = importlib.util.spec_from_file_location("check_publication", SCRIPT)
assert SPEC and SPEC.loader
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "--quiet")
    # The fixture must not inherit a developer's global ignored filenames.
    excludes = tmp_path / ".git" / "publication-empty-excludes"
    excludes.write_text("")
    git(tmp_path, "config", "core.excludesFile", str(excludes))
    return tmp_path


def stage(repo, name, text):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    git(repo, "add", "--force", "--", name)
    return path


def test_synthetic_fixture_code_and_placeholder_config_are_allowed(repo):
    stage(repo, "examples/voice-corpus/rowan-harbor.txt", "Invented fixture with no source author.")
    stage(repo, "src/example.py", 'API_KEY = "sk-test-placeholder"\n')
    stage(repo, "src/colophon/reporting/templates/report.html", "<p>{{ content }}</p>")
    stage(repo, ".env.example", "API_KEY=CHANGEME\n")
    assert publication.check(repo, "staged", []) == []
    assert publication.check(repo, "tracked", []) == []


@pytest.mark.parametrize("name", [
    "outputs/draft.md", "corpus/author.analysis.json", "openmemory.md", "books/book.epub",
    "draft.voice.json", "private.prompt.md", "profiles/author.json", "annotations/notes.json",
    "source-material/essay.html", "archive.txt", ".env.local", ".env.production",
    "service-account-private.json", "credentials.json", "key.pem", ".venv/cache.py",
])
def test_forced_added_private_artifact_is_blocked_by_path_without_reading(repo, name, monkeypatch):
    stage(repo, name, "Opaque private bytes")
    original = publication.git

    def no_blob_read(root, *args):
        assert args[0] != "show", "Forbidden file contents must never be opened"
        return original(root, *args)

    monkeypatch.setattr(publication, "git", no_blob_read)
    for mode in ("staged", "tracked", "worktree"):
        result = publication.check(repo, mode, [])
        assert len(result) == 1 and name in result[0] and "forbidden publication path" in result[0]


def test_staged_blob_is_checked_even_if_worktree_is_cleaned(repo):
    secret = "ghp_" + "Q7" * 20
    path = stage(repo, "src/example.py", secret)
    path.write_text("safe")
    result = publication.check(repo, "staged", [])
    assert len(result) == 1 and "GitHub token" in result[0]
    assert secret not in "\n".join(result)
    assert publication.check(repo, "worktree", []) == []


@pytest.mark.parametrize("private", [
    "/" + "Users" + "/invented-person/private/project",
    "/" + "home" + "/invented-person/private/project",
    "C:" + "\\Users" + "\\invented-person\\private",
    "AK" + "IA" + "9Z" * 8,
    "sk-" + "aZ7" * 15,
    "-----BEGIN " + "PRIVATE KEY-----",
    "xoxb-" + "12345-" * 6,
    "AIza" + "aZ7" * 11 + "ab",
])
def test_sensitive_patterns_report_path_line_but_never_the_value(private):
    result = publication.content_findings("src/example.py", ("safe\n" + private).encode())
    assert result and all(line.startswith("src/example.py:2:") for line in result)
    assert private not in "\n".join(result)


def test_deleted_staged_path_is_not_read(repo):
    stage(repo, "src/example.py", "safe")
    git(repo, "rm", "--cached", "src/example.py")
    assert publication.check(repo, "staged", []) == []


def test_symlink_target_is_not_opened(repo, monkeypatch):
    target = repo / ".env"
    target.write_text("opaque")
    (repo / "alias.py").symlink_to(target.name)
    git(repo, "add", "alias.py")

    def no_content_read(path):
        pytest.fail("Symlink and forbidden target bytes must not be opened")

    monkeypatch.setattr(Path, "read_bytes", no_content_read)
    tracked = publication.check(repo, "tracked", [])
    assert len(tracked) == 1 and tracked[0].startswith("alias.py:") and "symlink" in tracked[0]
    worktree = publication.check(repo, "worktree", [])
    assert len(worktree) == 2
    assert any(line.startswith("alias.py:") and "symlink" in line for line in worktree)
    assert any(line.startswith(".env:") and "forbidden publication path" in line for line in worktree)


def test_optional_overlap_reports_candidate_only(tmp_path):
    source = tmp_path / "ignored-source"
    source.mkdir()
    passage = (
        "Amber robots stack circular lanterns beside violet canals while quiet astronomers "
        "catalog distant moons beneath silver clouds before dawn arrives across the orchard."
    )
    (source / "private-essay.html").write_text("<article><p>" + passage + "</p></article>")
    result = publication.source_overlap({"docs/copied.md": passage.encode()}, [source])
    assert len(result) == 1 and "docs/copied.md" in result[0]
    assert "private-essay" not in result[0] and passage not in result[0]
    assert publication.source_overlap({"src/short.py": b"Amber robots stack"}, [source]) == []


def test_source_overlap_does_not_read_env_or_symlinks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / ".env.txt").write_text("opaque")
    (source / "link.txt").symlink_to(".env.txt")
    assert publication.source_overlap({"safe.py": b"safe"}, [source]) == []


def test_worktree_mode_checks_nonignored_untracked_candidates(repo):
    (repo / "fresh.py").write_text("/" + "Users" + "/invented-person/private/file")
    assert publication.check(repo, "tracked", []) == []
    assert "fresh.py:1:" in publication.check(repo, "worktree", [])[0]


def test_ignore_policy_preserves_only_synthetic_raw_text(repo):
    ignore = SCRIPT.parents[1] / ".gitignore"
    (repo / ".gitignore").write_bytes(ignore.read_bytes())
    assert git(repo, "check-ignore", "outputs/draft.md", "corpus/author.html", "openmemory.md",
               ".env.local", "author.prompt.md", "author.txt").count(b"\n") == 6
    result = subprocess.run(["git", "-C", str(repo), "check-ignore", "examples/voice-corpus/rowan-harbor.txt"],
                            capture_output=True)
    assert result.returncode == 1

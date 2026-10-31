#!/usr/bin/env python3
"""Check publication candidates without exposing matched content.

Default: staged additions/modifications, reading index blobs. --tracked checks all
index blobs (suitable for CI). --worktree checks tracked and nonignored untracked
files as they exist locally. Forbidden paths are rejected BEFORE reading bytes.
Optionally compare candidates with --source-overlap DIR (local, ignored material;
never needed by CI). This is a conservative guard, not copyright/secret clearance.
No files or index entries are changed. Diagnostics contain paths/line numbers only.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import re
import subprocess
from pathlib import Path, PurePosixPath

PRIVATE_DIRS = {
    "outputs", "corpus", "data", "books", "downloads", "source-material", "archives",
    "profiles", "prompts", "extracts", "annotations", "generated-essays", "node_modules",
    ".venv", "venv", "__pycache__", ".cache", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ".pyright", ".codex", ".claude", ".git", "build", "dist", "assets",
}
ARTIFACT_SUFFIXES = (
    ".voice.json", ".profile.json", ".analysis.json", ".colophon.json", ".prompt.txt",
    ".prompt.md", ".prompt.json", ".extract.txt", ".extract.json", ".annotations.json",
    ".restyled.md", ".report.html",
    ".epub", ".pdf", ".mobi", ".azw", ".azw3", ".pem", ".p12", ".key",
)
# Deliberately high-specificity signatures; examples such as sk-test/CHANGEME are safe.
CONTENT_RULES = {
    "private absolute path": re.compile(r"(?:/Users/|/home/)[A-Za-z0-9_.-]+/|[A-Za-z]:\\Users\\[^\\\s]+\\"),
    "private key material": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "provider API key": re.compile(r"\bsk-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{32,}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "Google API key": re.compile(r"\bAIza[A-Za-z0-9_-]{35}\b"),
}


def forbidden_path(name: str) -> str | None:
    path = PurePosixPath(name)
    parts = tuple(part.lower() for part in path.parts)
    base = path.name.lower()
    synthetic = len(parts) == 3 and parts[:2] == ("examples", "voice-corpus")
    if path.is_absolute() or ".." in parts:
        return "path must be repository relative"
    if any(part in PRIVATE_DIRS for part in parts[:-1]) and not synthetic:
        return "local corpus, output, private artifact or cache directory"
    if base in {"openmemory.md", "claude.md", ".ds_store",
                "profile.json", "profiles.json", "annotations.json", "prompt.txt", "prompt.md"}:
        return "private local note or generated artifact"
    if name == "report.html":
        return "generated report"
    if base.startswith(".env") and base not in {".env.example", ".env.sample"}:
        return "environment file (contents not read)"
    if base.startswith(("credentials", "service-account", "id_rsa", "id_ed25519")):
        return "credential file (contents not read)"
    if base.endswith(ARTIFACT_SUFFIXES):
        return "source book, credential or derived author artifact"
    if base.endswith(".txt") and not synthetic and not base.startswith("requirements"):
        return "raw text outside the explicitly synthetic fixture directory"
    return None


def content_findings(name: str, data: bytes) -> list[str]:
    text = data.decode("utf-8", errors="replace")
    findings = []
    for label, pattern in CONTENT_RULES.items():
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            findings.append(f"{name}:{line}: {label}; remove private content before publication")
    return findings


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)


def candidates(root: Path, mode: str) -> list[str]:
    if mode == "staged":
        raw = git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    elif mode == "tracked":
        raw = git(root, "ls-files", "-z")
    else:
        raw = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return sorted(set(raw.decode("utf-8", errors="surrogateescape").rstrip("\0").split("\0")) - {""})


def windows(text: str, size: int = 20):
    """Distinctive normalized exact word runs; HTML tags/entities are normalized."""
    clean = html.unescape(re.sub(r"<[^>]*>", " ", text))
    words = list(re.finditer(r"\b[\w'-]+\b", clean.lower()))
    for start in range(len(words) - size + 1):
        run = [word.group() for word in words[start:start + size]]
        if len(set(run)) >= 10:
            yield hashlib.sha256(" ".join(run).encode()).digest()


def source_overlap(blobs: dict[str, bytes], directories: list[Path]) -> list[str]:
    """Compare candidate fingerprints; never print passages or private source names."""
    fingerprints: dict[bytes, set[str]] = {}
    for name, data in blobs.items():
        for fingerprint in windows(data.decode("utf-8", errors="replace")):
            fingerprints.setdefault(fingerprint, set()).add(name)
    matched: set[str] = set()
    for directory in directories:
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("source-overlap requires an existing nonsymlink directory")
        for path in directory.rglob("*"):
            if path.suffix.lower() not in {".txt", ".md", ".html", ".htm"}:
                continue
            # Never follow a link to private files outside the selected source tree.
            if any(parent.is_symlink() for parent in (path, *path.parents)):
                continue
            if any(part.startswith(".env") for part in path.parts) or not path.is_file():
                continue
            for fingerprint in windows(path.read_text(errors="replace")):
                matched.update(fingerprints.get(fingerprint, ()))
    return [f"{name}: distinctive 20-word overlap with local source; review/remove copied material"
            for name in sorted(matched)]


def check(root: Path, mode: str, sources: list[Path]) -> list[str]:
    findings: list[str] = []
    blobs: dict[str, bytes] = {}
    for name in candidates(root, mode):
        reason = forbidden_path(name)
        if reason:
            findings.append(f"{name}: forbidden publication path: {reason}; keep local and untrack it")
            continue
        if mode == "worktree":
            path = root / name
            if not path.exists():
                continue
            if path.is_symlink():
                findings.append(f"{name}: symlink requires manual publication review (target not read)")
                continue
            data = path.read_bytes()
        else:
            # Symlink targets in index are bytes, never followed; reject them as well.
            entry = git(root, "ls-files", "--stage", "--", name)
            if entry.startswith(b"120000 "):
                findings.append(f"{name}: symlink requires manual publication review (target not read)")
                continue
            data = git(root, "show", f":{name}")
        findings.extend(content_findings(name, data))
        if sources:
            blobs[name] = data
    if sources:
        findings.extend(source_overlap(blobs, sources))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--tracked", action="store_true", help="check every indexed blob, for CI")
    modes.add_argument("--worktree", action="store_true", help="check tracked and nonignored local candidates")
    parser.add_argument("--source-overlap", action="append", type=Path, default=[], metavar="DIR")
    args = parser.parse_args()
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
    mode = "tracked" if args.tracked else "worktree" if args.worktree else "staged"
    try:
        findings = check(root, mode, args.source_overlap)
    except (OSError, ValueError, subprocess.CalledProcessError):
        print("Publication check could not complete; verify Git state and local source access.")
        return 2
    for finding in findings:
        print(finding)
    if findings:
        print("Publication check failed. No files were deleted or unstaged.")
        return 1
    print("Publication checks passed for selected candidates; manual rights and privacy review is still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

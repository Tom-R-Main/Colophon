"""Read-only, private local aliases for portable VoiceProfile files.

Nothing is discovered or read until an explicit list/resolve call. Git is used
only to inspect local tracking and ignore metadata; no source corpus is opened.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections import deque
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from colophon.voice.models import VoiceProfile

_ALIAS = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_RECOVERY = "Use an absolute local file outside Git, or an untracked Git-ignored file; keep private files uncommitted."


class RegistryError(ValueError):
    """A stable, content-free diagnostic suitable for machine-readable output."""

    def __init__(self, code: str, message: str, recovery: str) -> None:
        self.code = code
        self.message = message
        self.recovery = recovery
        super().__init__(message)


def _error(code: str) -> RegistryError:
    messages = {
        "registry_not_found": "No local profile registry was found.",
        "invalid_registry": "The local profile registry has an invalid path, JSON, or schema.",
        "registry_not_private": "The local profile registry is not in an allowed private location.",
        "unknown_profile": "The requested profile alias is not registered.",
        "profile_not_found": "The selected local profile file was not found.",
        "profile_not_private": "The selected local profile is not in an allowed private location.",
        "invalid_profile": "The selected file is not a supported, valid VoiceProfile.",
        "privacy_check_failed": "Local file privacy could not be verified.",
    }
    recovery = _RECOVERY
    if code in {"registry_not_found", "invalid_registry"}:
        recovery += " Configure a schema_version 1 registry with named VoiceProfile paths."
    elif code == "unknown_profile":
        recovery = "List the configured profile aliases and select an existing name."
    elif code == "invalid_profile":
        recovery = "Use voice extract to build a VoiceProfile with schema_version and extractor_version both '1'."
    elif code == "privacy_check_failed":
        recovery = (
            "Check file access, local Git metadata and safe file-open support; "
            "retry in a supported environment without disabling privacy checks."
        )
    return RegistryError(code, messages[code], recovery)


def _absolute(value: str | Path, code: str) -> Path:
    try:
        path = Path(value).expanduser()
        # No interpolation, shell patterns, or cwd-relative references.
        if not path.is_absolute() or any(character in str(path) for character in ("$", "*", "?", "[", "\x00")):
            raise _error(code)
        return path
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        if isinstance(error, RegistryError):
            raise
        raise _error(code) from None


def _registry_path(registry: Path | None) -> Path:
    if registry is not None:
        return _absolute(registry, "invalid_registry")
    if "COLOPHON_PROFILES_FILE" in os.environ:
        return _absolute(os.environ["COLOPHON_PROFILES_FILE"], "invalid_registry")
    home = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return _absolute(home, "invalid_registry") / "colophon" / "profiles.json"


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    # Inherited Git overrides must not redirect repository discovery or indexes.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    try:
        return subprocess.run(
            [
                "git", "--no-optional-locks",
                "-c", "core.fsmonitor=false",
                "-C", str(root), *arguments,
            ],
            capture_output=True,
            check=False,
            timeout=5,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise _error("privacy_check_failed") from None


def _tracked_file(path: Path, root: Path) -> bool:
    relative = str(path.relative_to(root))
    # Git's index matching is case-sensitive even on a case-insensitive volume.
    # Query only this name's candidates; an icase match is not proof of identity
    # on Linux, where two case-different names can be separate private files.
    tracked = _git(
        root, "ls-files", "--cached", "-z", "--", f":(literal){relative}", f":(icase,literal){relative}"
    )
    if tracked.returncode != 0 or (tracked.stdout and not tracked.stdout.endswith(b"\0")):
        raise _error("privacy_check_failed")
    try:
        selected = path.lstat()
        for raw in tracked.stdout.split(b"\0")[:-1]:
            candidate = Path(os.fsdecode(raw))
            if candidate.is_absolute() or ".." in candidate.parts:
                raise _error("privacy_check_failed")
            if str(candidate) == relative:
                return True
            try:
                identity = (root / candidate).lstat()
            except FileNotFoundError:
                # A stale, differently-cased index entry has no local identity.
                continue
            # lstat compares link identities as well as regular file identities;
            # following both links would confuse distinct links to one target.
            if os.path.samestat(selected, identity):
                return True
        return False
    except OSError:
        raise _error("privacy_check_failed") from None


def _check_location(path: Path, code: str) -> None:
    if any(part.casefold() == ".git" for part in path.parts):
        raise _error(code)
    # Inspect every enclosing checkout, including an outer checkout around a
    # nested repository. A missing/unreadable marker is never assumed harmless.
    for parent in path.parents:
        marker = parent / ".git"
        try:
            marker.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise _error("privacy_check_failed") from None
        relative = str(path.relative_to(parent))
        if _tracked_file(path, parent):
            raise _error(code)
        ignored = _git(parent, "check-ignore", "--no-index", "-q", "--", relative)
        if ignored.returncode == 1:
            raise _error(code)
        if ignored.returncode != 0:
            raise _error("privacy_check_failed")


def _private_path(path: Path, code: str) -> Path:
    try:
        if any(part.casefold() == ".git" for part in path.parts):
            raise _error(code)
        # Resolve one component and one link at a time. Path.resolve() collapses
        # entire chains, hiding intermediate links that may themselves be tracked.
        # Keep '..' pending until all preceding symlinks have been honored.
        pending = deque(path.parts[1:])
        resolved = Path(path.anchor)
        links = 0
        while pending:
            part = pending.popleft()
            if part == "..":
                resolved = resolved.parent
                continue
            if part.casefold() == ".git":
                raise _error(code)
            component = resolved / part
            if stat.S_ISLNK(component.lstat().st_mode):
                _check_location(component, code)
                links += 1
                if links > 40:
                    raise _error("privacy_check_failed")
                target = component.readlink()
                if target.is_absolute():
                    resolved = Path(target.anchor)
                    pending.extendleft(reversed(target.parts[1:]))
                else:
                    pending.extendleft(reversed(target.parts))
            else:
                resolved = component
        _check_location(path, code)
        _check_location(resolved, code)
        return resolved
    except RegistryError:
        raise
    except FileNotFoundError:
        raise _error(code.removesuffix("_not_private") + "_not_found") from None
    except (OSError, RuntimeError, ValueError):
        raise _error("privacy_check_failed") from None


def _read_private(path: Path, kind: str) -> bytes:
    # A nonexistent file has no contents to expose. Report missing configuration
    # consistently even when its future destination is inside a checkout.
    try:
        path.stat()
    except FileNotFoundError:
        raise _error(f"{kind}_not_found") from None
    except (OSError, ValueError):
        raise _error("privacy_check_failed") from None
    resolved = _private_path(path, f"{kind}_not_private")
    try:
        # Avoid following a replaced final symlink or blocking on a FIFO.
        nofollow = getattr(os, "O_NOFOLLOW", None)
        nonblock = getattr(os, "O_NONBLOCK", None)
        if not nofollow or not nonblock:
            raise _error("privacy_check_failed")
        descriptor = os.open(resolved, os.O_RDONLY | nonblock | nofollow)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            os.close(descriptor)
            raise _error(f"invalid_{kind}")
        with os.fdopen(descriptor, "rb") as stream:
            confirmed = _private_path(path, f"{kind}_not_private")
            current = confirmed.stat()
            if confirmed != resolved or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
                raise _error("privacy_check_failed")
            return stream.read()
    except FileNotFoundError:
        raise _error(f"{kind}_not_found") from None
    except OSError:
        raise _error("privacy_check_failed") from None


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(_: str) -> None:
    raise ValueError("Non-finite JSON number")


def _json(source: bytes, code: str) -> Any:
    try:
        return json.loads(source, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (ValueError, UnicodeError, RecursionError):
        raise _error(code) from None


def _entries(path: Path) -> dict[str, Path]:
    data = _json(_read_private(path, "registry"), "invalid_registry")
    if (
        not isinstance(data, dict)
        or set(data) != {"schema_version", "profiles"}
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or not isinstance(data["profiles"], dict)
    ):
        raise _error("invalid_registry")
    entries = {}
    for name, entry in data["profiles"].items():
        if (
            not _ALIAS.fullmatch(name)
            or not isinstance(entry, dict)
            or set(entry) != {"path"}
            or not isinstance(entry["path"], str)
        ):
            raise _error("invalid_registry")
        entries[name] = _absolute(entry["path"], "invalid_registry")
    return entries


def list_profiles(registry: Path | None = None) -> dict[str, Any]:
    """List alias metadata only; profile files and source documents stay unopened."""
    path = _registry_path(registry)
    try:
        entries = _entries(path)
    except RegistryError as error:
        if error.code != "registry_not_found":
            raise
        return {
            "schema_version": 1,
            "status": "not_configured",
            "registry_path": str(path),
            "profiles": [],
            "recovery": error.recovery,
        }
    return {
        "schema_version": 1,
        "status": "configured",
        "registry_path": str(path),
        "profiles": [{"name": name, "kind": "VoiceProfile"} for name in sorted(entries)],
        "validation_scope": "registry entries only",
    }


def resolve_profile(name: str, registry: Path | None = None) -> dict[str, Any]:
    """Validate one selected profile and return its local identity, never prose."""
    path = _registry_path(registry)
    entries = _entries(path)
    if not isinstance(name, str) or not _ALIAS.fullmatch(name) or name not in entries:
        raise _error("unknown_profile")
    profile_path = entries[name]
    source = _read_private(profile_path, "profile")
    data = _json(source, "invalid_profile")
    if not isinstance(data, dict) or data.get("schema_version") != "1" or data.get("extractor_version") != "1":
        raise _error("invalid_profile")
    try:
        profile = VoiceProfile.model_validate_json(source, strict=True)
    except (ValidationError, ValueError, RecursionError):
        raise _error("invalid_profile") from None
    return {
        "schema_version": 1,
        "status": "resolved",
        "name": name,
        "kind": "VoiceProfile",
        "registry_path": str(path),
        "profile_path": str(_private_path(profile_path, "profile_not_private")),
        "profile_id": profile.profile_id,
        "profile_sha256": hashlib.sha256(source).hexdigest(),
        "language": profile.language,
        "documents": len(profile.documents),
        "validation_scope": "profile schema and local file identity",
    }

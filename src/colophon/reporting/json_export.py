"""JSON export for analysis results."""

from __future__ import annotations

from pathlib import Path

from colophon.models.features import StyleProfile


def export_json(profile: StyleProfile, output: Path) -> None:
    """Export a StyleProfile to JSON."""
    output.write_text(profile.model_dump_json(indent=2))

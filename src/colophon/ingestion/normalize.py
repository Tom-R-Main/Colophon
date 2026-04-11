"""Text normalization applied after all ingesters."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize extracted text for consistent analysis."""
    text = unicodedata.normalize("NFC", text)
    text = _fix_ligatures(text)
    text = _normalize_quotes(text)
    text = _normalize_dashes(text)
    text = _collapse_whitespace(text)
    return text.strip()


def _fix_ligatures(text: str) -> str:
    """Replace common OCR ligature artifacts."""
    replacements = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb00": "ff",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _normalize_quotes(text: str) -> str:
    """Normalize smart quotes to straight quotes."""
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return text


def _normalize_dashes(text: str) -> str:
    """Normalize various dash characters but preserve em-dashes for punctuation analysis."""
    text = text.replace("\u2013", "--")   # en-dash -> double hyphen
    text = text.replace("\u2014", "---")  # em-dash -> triple hyphen
    return text


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace while preserving paragraph breaks."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs (not newlines) to single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Strip trailing whitespace per line
    text = re.sub(r" +\n", "\n", text)
    return text

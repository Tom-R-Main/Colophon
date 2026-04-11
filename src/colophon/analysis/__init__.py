"""Stylometric analysis pipeline."""

from __future__ import annotations

from typing import Any


def load_spacy_model(model_name: str = "en_core_web_sm") -> Any:
    """Load a spaCy model with a helpful error if not installed."""
    try:
        import spacy
        return spacy.load(model_name)
    except OSError:
        raise RuntimeError(
            f"spaCy model '{model_name}' not found. Install it with:\n"
            f"  python -m spacy download {model_name}"
        )
    except ImportError:
        raise RuntimeError(
            "spaCy is not installed. Install it with:\n"
            "  pip install colophon[analysis]"
        )

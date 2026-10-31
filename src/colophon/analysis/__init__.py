"""Stylometric analysis pipeline."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=2)
def load_spacy_model(model_name: str = "en_core_web_sm") -> Any:
    """Reuse at most two local linguistic pipelines; parsed documents are not cached.

    Call ``load_spacy_model.cache_clear()`` after changing an installed model in
    a long-lived process. Measurements record the actual loaded model metadata.
    """
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

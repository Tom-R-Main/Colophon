"""OCR preprocessing for scanned PDFs."""

from __future__ import annotations

import tempfile
from pathlib import Path


def ocr_pdf(path: Path) -> Path:
    """Run OCR on a scanned PDF and return the path to the OCR'd version."""
    try:
        import ocrmypdf
    except ImportError:
        raise RuntimeError(
            "OCR support requires ocrmypdf. Install it with:\n"
            "  pip install colophon[ocr]\n"
            "Also requires system Tesseract: brew install tesseract"
        )

    output = Path(tempfile.mktemp(suffix=".pdf"))
    ocrmypdf.ocr(
        str(path),
        str(output),
        deskew=True,
        clean=True,
        skip_text=True,  # Don't re-OCR pages that already have text
    )
    return output

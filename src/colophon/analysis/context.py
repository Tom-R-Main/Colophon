"""One parsed document and a shared lexical-token counting contract.

A word unit is a non-space, non-punctuation spaCy token containing a letter
or number. Tokenizer-split
contractions count as multiple units; punctuation never changes denominators.
"""

from __future__ import annotations

import re
from typing import Any


def lexical_tokens(doc: Any) -> list[Any]:
    """Return the lexical units shared by word-based analyzers."""
    return [
        token for token in doc
        if not token.is_space and not token.is_punct and any(character.isalnum() for character in token.text)
    ]


class AnalysisContext:
    """Adapt the existing analyzer API to a single authoritative parse."""

    def __init__(
        self, text: str, nlp: Any, *, paragraph_boundaries: bool = True,
        paragraph_ranges: list[tuple[int, int]] | None = None,
    ) -> None:
        self.text = text
        # Explicit reviewed block ranges are authoritative, including an empty
        # list. Only an absent range list permits legacy blank-line inference.
        ranges = paragraph_ranges if paragraph_ranges is not None else [
            match.span() for match in re.finditer(r"\S[\s\S]*?(?=\n[ \t]*\n|\Z)", text)
        ]
        self.paragraph_ranges = []
        previous_end = 0
        for start, end in ranges:
            if start < previous_end or end <= start or start < 0 or end > len(text):
                raise ValueError("Paragraph ranges must be ordered, nonoverlapping text ranges")
            previous_end = end
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start < end:
                self.paragraph_ranges.append((start, end))
        doc = nlp.make_doc(text)
        if paragraph_boundaries:
            # A qualified logical paragraph cannot share a sentence with its neighbor.
            # Set starts before the parser rather than changing parsed dependencies.
            for start, end in self.paragraph_ranges:
                span = doc.char_span(start, end, alignment_mode="strict")
                if span is None:
                    raise ValueError("Paragraph ranges must align with tokenizer token boundaries")
                words = lexical_tokens(span)
                if words:
                    words[0].is_sent_start = True
        self.doc = nlp(doc)

    def __call__(self, text: str) -> Any:
        if text != self.text:
            raise ValueError("Analyzer requested text outside its shared analysis context")
        return self.doc

    def paragraph_spans(self) -> list[Any]:
        """Project authoritative logical ranges onto the shared token stream."""
        spans = []
        for start, end in self.paragraph_ranges:
            span = self.doc.char_span(start, end, alignment_mode="strict")
            if span is None:
                raise ValueError("Paragraph ranges must align with tokenizer token boundaries")
            if lexical_tokens(span):
                spans.append(span)
        return spans

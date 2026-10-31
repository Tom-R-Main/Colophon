"""Conservative structural and lexical observations with exact source spans.

English markers are literal signals, not classifiers of intention or personality.
Structural observations use normalized paragraph boundaries supplied by ingestion.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, cast

from colophon.models.document import DocumentBlock, text_blocks
from colophon.voice.models import EvidenceSpan, VoiceTrait

if TYPE_CHECKING:
    from colophon.voice.manifest import LoadedDocument

WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
Category = Literal["openings", "endings", "argument", "stance", "rhythm", "transitions", "quotations"]
CONTRAST = re.compile(r"\b(?:but|however|yet|although|whereas|instead|nevertheless)\b", re.I)
CAUSE = re.compile(r"\b(?:because|therefore|consequently|thus)\b", re.I)
FIRST_PERSON = re.compile(r"\b(?:i|me|my|mine|we|us|our|ours)\b", re.I)
ADDRESS = re.compile(r"\b(?:you|your|yours)\b", re.I)
HEDGE = re.compile(r"\b(?:perhaps|maybe|might|apparently|seems|possibly)\b", re.I)
TRANSITION = re.compile(r"^(?:however|meanwhile|instead|nevertheless|first|finally|therefore|for example)\b", re.I)


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def rhetoric_features(
    text: str,
    language: str = "en",
    *,
    source_text: str | None = None,
    blocks: list[DocumentBlock] | None = None,
) -> dict[str, float]:
    """Transparent marker rates for evaluation, without rhetorical-intent claims."""
    original = source_text if source_text is not None else text
    if len(original) != len(text):
        raise ValueError("Source and author text must share normalized offsets")
    source_blocks = text_blocks(original) if blocks is None else blocks
    ranges = [
        (block.char_offset_start, block.char_offset_end)
        for block in source_blocks
        if block.kind == "paragraph" and block.attribution == "author"
    ]
    complete = [(start, end) for start, end in ranges if original[start:end] == text[start:end]]
    parts = [original[start:end] for start, end in complete]
    total = max(len(WORD.findall(text)), 1)
    features = {}
    if parts:
        features["short_paragraph"] = sum(len(WORD.findall(p)) <= 8 for p in parts) / len(parts)
    if ranges and ranges[0] in complete:
        features["opening_question"] = float("?" in original[slice(*ranges[0])])
    if ranges and ranges[-1] in complete:
        ending = original[slice(*ranges[-1])]
        features["ending_question"] = float("?" in ending)
        features["short_ending"] = float(len(WORD.findall(ending)) <= 12)
    if language == "en":
        for key, pattern in (
            ("contrast_markers", CONTRAST),
            ("cause_markers", CAUSE),
            ("first_person", FIRST_PERSON),
            ("reader_address", ADDRESS),
            ("hedge_markers", HEDGE),
        ):
            features[key] = len(pattern.findall(text)) * 1000 / total
        if parts:
            features["transition_openers"] = sum(bool(TRANSITION.search(p)) for p in parts) / len(parts)
    return features


def author_evidence(
    doc: LoadedDocument, *, role: Literal["support", "counterexample", "example"] = "support"
) -> list[EvidenceSpan]:
    """Return included source regions; never bridge masked exclusions."""
    source = doc.document.text
    author = doc.author_document.text
    result = []
    blocks = getattr(doc.author_document, "blocks", [])
    for block in blocks:
        if block.kind not in {"paragraph", "unknown"} or block.attribution != "author":
            continue
        start = block.char_offset_start
        end = block.char_offset_end
        # Exclusions can occur inside a block. Split at changed characters so
        # no excerpt silently reintroduces a quotation or another contributor.
        run_start = start
        for index in range(start, end + 1):
            changed = index == end or source[index] != author[index]
            if not changed:
                continue
            raw = source[run_start:index]
            trimmed = raw.strip()
            if trimmed:
                offset = run_start + len(raw) - len(raw.lstrip())
                result.append(
                    EvidenceSpan(
                        document_id=doc.entry.id,
                        source_hash=doc.normalized_hash,
                        block_id=block.id,
                        start=offset,
                        end=offset + len(trimmed),
                        text=trimmed,
                        role=role,
                    )
                )
            run_start = index + 1
    return result


def _structural_spans(doc: LoadedDocument, spans: list[EvidenceSpan], category: str) -> list[EvidenceSpan]:
    """Whole logical paragraphs only; fragments cannot become openings/endings."""
    if doc.author_document.metadata.get("paragraph_quality") == "unverified":
        return []
    original_blocks = [
        block
        for block in doc.document.blocks
        if block.kind in {"paragraph", "unknown"} and block.attribution == "author"
    ]
    original_by_id = {block.id: block for block in original_blocks}
    by_block = {}
    for span in spans:
        block = original_by_id.get(span.block_id)
        if block is not None and block.char_offset_start == span.start and block.char_offset_end == span.end:
            by_block[span.block_id] = span
    if category in {"openings", "endings"}:
        if not original_blocks:
            return []
        edge = original_blocks[0] if category == "openings" else original_blocks[-1]
        return [by_block[edge.id]] if edge.id in by_block else []
    return [by_block[block.id] for block in original_blocks if block.id in by_block]


def extract_traits(loaded: list[LoadedDocument]) -> list[VoiceTrait]:
    """Describe recurring observable patterns and retain contrasting documents."""
    spans = {doc.entry.id: author_evidence(doc) for doc in loaded}
    usable = [doc for doc in loaded if spans[doc.entry.id]]
    if not usable:
        return []
    english = all(doc.entry.language == "en" for doc in usable)
    rules: list[tuple[str, str, str, Callable[[list[EvidenceSpan]], EvidenceSpan | None]]] = [
        (
            "opening-question",
            "openings",
            "A question mark occurs in the opening author paragraph",
            lambda items: items[0] if "?" in items[0].text else None,
        ),
        (
            "ending-question",
            "endings",
            "A question mark occurs in the closing author paragraph",
            lambda items: items[-1] if "?" in items[-1].text else None,
        ),
        (
            "short-ending",
            "endings",
            "The closing author paragraph contains at most twelve lexical words",
            lambda items: items[-1] if len(WORD.findall(items[-1].text)) <= 12 else None,
        ),
        (
            "short-paragraph",
            "rhythm",
            "An author paragraph contains at most eight lexical words",
            lambda items: next((p for p in items if len(WORD.findall(p.text)) <= 8), None),
        ),
    ]
    if english:
        for trait_id, category, statement, pattern in (
            (
                "contrast-markers",
                "argument",
                "An explicit contrast marker (but, however, yet, although, whereas, instead or nevertheless) occurs",
                CONTRAST,
            ),
            (
                "cause-markers",
                "argument",
                "An explicit cause or consequence marker (because, therefore, consequently or thus) occurs",
                CAUSE,
            ),
            ("first-person", "stance", "A first-person English pronoun occurs in author narration", FIRST_PERSON),
            ("reader-address", "stance", "A second-person English pronoun occurs in author narration", ADDRESS),
            (
                "hedge-markers",
                "stance",
                "A listed uncertainty marker (perhaps, maybe, might, apparently, seems or possibly) occurs",
                HEDGE,
            ),
            (
                "transition-openers",
                "transitions",
                "An author paragraph starts with a listed transition marker",
                TRANSITION,
            ),
        ):
            rules.append(
                (
                    trait_id,
                    category,
                    statement,
                    lambda items, pattern=pattern: next((p for p in items if pattern.search(p.text)), None),
                )
            )

    traits = []
    for trait_id, category, statement, match in rules:
        support = []
        counterexamples = []
        structural = category in {"openings", "endings", "rhythm", "transitions"}
        inputs = {
            doc.entry.id: _structural_spans(doc, spans[doc.entry.id], category) if structural else spans[doc.entry.id]
            for doc in usable
        }
        eligible = [doc for doc in usable if inputs[doc.entry.id]]
        if not eligible:
            continue
        for doc in eligible:
            items = inputs[doc.entry.id]
            found = match(items)
            if found:
                support.append(found)
            else:
                counterexamples.append(items[0].model_copy(update={"role": "counterexample"}))
        if not support:
            continue
        supporting_ids = {item.document_id for item in support}
        works = {doc.entry.work_id for doc in usable if doc.entry.id in supporting_ids}
        traits.append(
            VoiceTrait(
                id=trait_id,
                category=cast(Category, category),
                statement=f"{statement} in {len(support)} of {len(eligible)} eligible independent pieces.",
                status="observed" if len(works) >= 2 else "tentative",
                evidence=support,
                counterexamples=counterexamples,
                exceptions=[
                    "Literal marker/structure observation; rhetorical intent and distinctiveness are not inferred."
                ]
                + (["English marker rules only."] if category in {"argument", "stance", "transitions"} else [])
                + (
                    [
                        "Only complete included author paragraphs are eligible; partially excluded edge paragraphs "
                        "do not establish openings or endings."
                    ]
                    if structural
                    else []
                )
                + (["Only one independent work supports this pattern."] if len(works) < 2 else []),
            )
        )

    quote_support = []
    quote_counterexamples = []
    for doc in usable:
        blocks = getattr(doc.document, "blocks", [])
        quotes = [i for i, b in enumerate(blocks) if b.kind == "quotation" or b.attribution == "quoted"]
        found = None
        for index in quotes:
            adjacent = {blocks[i].id for i in (index - 1, index + 1) if 0 <= i < len(blocks)}
            found = next((p for p in spans[doc.entry.id] if p.block_id in adjacent), None)
            if found:
                break
        if found is None:
            from colophon.voice.manifest import quoted_ranges

            ranges, _ = quoted_ranges(doc.document.text)
            shared_blocks = {
                block.id
                for block in blocks
                for start, end in ranges
                if block.char_offset_start <= start < end <= block.char_offset_end
            }
            found = next((span for span in spans[doc.entry.id] if span.block_id in shared_blocks), None)
        if found:
            quote_support.append(found)
        else:
            quote_counterexamples.append(spans[doc.entry.id][0].model_copy(update={"role": "counterexample"}))
    if quote_support:
        traits.append(
            VoiceTrait(
                id="quote-adjacency",
                category="quotations",
                statement="Author narration borders a quotation block or shares a paragraph "
                "with paired quotation marks "
                f"in {len(quote_support)} of {len(usable)} independent pieces.",
                status="observed" if len(quote_support) >= 2 else "tentative",
                evidence=quote_support,
                counterexamples=quote_counterexamples,
                exceptions=[
                    "Adjacency describes layout; it does not establish agreement, "
                    "commentary intent or quotation accuracy."
                ],
            )
        )
    return traits

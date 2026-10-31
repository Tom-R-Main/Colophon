from pathlib import Path

import pytest

from colophon.ingestion import ingest
from colophon.models.document import Document, text_blocks
from colophon.voice.manifest import load_corpus


def write_manifest(tmp_path, entries):
    import json

    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"name": "Fixture corpus", "documents": entries}))
    return path


def entry(name, **kwargs):
    return {"id": name, "path": name + ".txt", "author": "Writer", "work_id": name, **kwargs}


def test_short_paragraphs_and_offsets_survive_normalization(tmp_path):
    path = tmp_path / "voice.txt"
    path.write_text("No.\r\n\r\nNever.\r\n\r\nA longer paragraph wraps\r\nonto another line.")
    document = ingest(path)
    assert [block.text for block in document.blocks] == [
        "No.",
        "Never.",
        "A longer paragraph wraps\nonto another line.",
    ]
    for block in document.blocks:
        assert document.text[block.char_offset_start : block.char_offset_end] == block.text
    assert [b.text for b in text_blocks(" a \n\n b\nwrapped.\n\n")] == ["a", "b\nwrapped."]


def test_epub_preserves_real_paragraphs_and_source_item(tmp_path):
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("fixture")
    book.set_title("Paragraphs")
    book.set_language("en")
    chapter = epub.EpubHtml(title="One", file_name="chapter.xhtml", lang="en")
    chapter.content = "<h1>Title</h1><p>No.</p><p>This paragraph wraps\ninside its XHTML element.</p>"
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = [chapter]
    path = tmp_path / "fixture.epub"
    epub.write_epub(str(path), book)
    document = ingest(path)
    paragraphs = [block for block in document.blocks if block.kind == "paragraph"]
    assert [block.text for block in paragraphs] == ["No.", "This paragraph wraps inside its XHTML element."]
    assert all(block.source_item for block in paragraphs)
    assert document.metadata["paragraph_quality"] == "explicit"
    assert "No.\n\nThis" in document.text


def test_pdf_retains_regions_and_reports_uncertain_prose_structure(tmp_path):
    import pymupdf as fitz

    path = tmp_path / "fixture.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "The first physical text block is longer than fifty characters.")
        page.insert_text((72, 140), "The second physical block is also long enough to avoid OCR.")
        pdf.save(path)
    document = ingest(path)
    assert document.blocks
    assert all(block.page == 1 and block.bbox for block in document.blocks)
    assert all(block.kind == "unknown" and block.provenance_quality == "region_only" for block in document.blocks)
    assert document.metadata["paragraph_quality"] == "unverified"
    assert document.warnings


def test_excluded_quotations_and_editor_spans_keep_original_offsets(tmp_path):
    text = 'EDITOR\n\nI watched. "Another person spoke." Then I left.'
    (tmp_path / "a.txt").write_text(text)
    path = write_manifest(tmp_path, [entry("a", excluded_spans=[{"start": 0, "end": 6, "reason": "editor heading"}])])
    loaded = load_corpus(path)[0]
    assert loaded.document.text == text
    assert len(loaded.author_document.text) == len(text)
    assert "Another person" not in loaded.author_document.text
    assert "EDITOR" not in loaded.author_document.text
    assert "Then I left." in loaded.author_document.text
    assert loaded.excluded_word_count > 0


def test_duplicates_and_works_cannot_leak_into_holdout(tmp_path):
    (tmp_path / "a.txt").write_text("A writer watches the river and thinks about the boats.")
    (tmp_path / "b.txt").write_text("A writer watches the river and thinks about the boats.")
    path = write_manifest(tmp_path, [entry("a"), entry("b", split="test")])
    with pytest.raises(ValueError, match="Duplicate text"):
        load_corpus(path)
    path = write_manifest(tmp_path, [entry("a"), entry("b", split="test", work_id="a")])
    with pytest.raises(ValueError, match="crosses"):
        load_corpus(path)


def test_translator_is_explicit_voice_owner(tmp_path):
    (tmp_path / "a.txt").write_text("This translated wording belongs to the translator's linguistic choices.")
    translated = entry("a", contributors=[{"name": "Translator", "role": "translator"}])
    path = write_manifest(tmp_path, [translated])
    with pytest.raises(ValueError, match="voice_author"):
        load_corpus(path)
    path = write_manifest(tmp_path, [{**translated, "voice_author": "Translator"}])
    assert load_corpus(path)[0].entry.effective_author == "Translator"


def test_sources_are_confined_and_exclusions_bounded(tmp_path):
    path = write_manifest(tmp_path, [entry("a", path="../outside.txt")])
    with pytest.raises(ValueError, match="inside"):
        load_corpus(path)
    (tmp_path / "a.txt").write_text("Words.")
    path = write_manifest(tmp_path, [entry("a", excluded_spans=[{"start": 0, "end": 900, "reason": "bad offset"}])])
    with pytest.raises(ValueError, match="exceeds"):
        load_corpus(path)


def test_blocks_reject_invalid_source_coordinates():
    blocks = text_blocks("One paragraph.")
    blocks[0] = blocks[0].model_copy(update={"text": "A different paragraph."})
    with pytest.raises(ValueError, match="exact text slices"):
        Document.from_text("One paragraph.", Path("test.txt"), blocks=blocks)


def test_excluded_heading_quote_cannot_consume_author_body(tmp_path):
    document = Document.from_text('A heading with an unmatched "\n\nI stayed to observe the room.', "fixture")
    blocks = [document.blocks[0].model_copy(update={"kind": "heading"}), document.blocks[1]]
    document = document.model_copy(update={"blocks": blocks})
    (tmp_path / "a.colophon.json").write_text(document.model_dump_json())
    path = write_manifest(tmp_path, [entry("a", path="a.colophon.json")])
    loaded = load_corpus(path)[0]
    assert "I stayed to observe the room." in loaded.author_document.text
    assert not any("Unclosed quotation" in warning for warning in loaded.warnings)


def test_explicit_quote_policy_preserves_author_terms(tmp_path):
    document = Document.from_text('By "good" I mean useful.\n\nSomeone else supplied this quotation.', "fixture")
    blocks = [document.blocks[0], document.blocks[1].model_copy(update={"kind": "quotation", "attribution": "quoted"})]
    document = document.model_copy(update={"blocks": blocks})
    (tmp_path / "a.colophon.json").write_text(document.model_dump_json())
    path = write_manifest(tmp_path, [entry("a", path="a.colophon.json", quotation_policy="explicit_only")])
    loaded = load_corpus(path)[0]
    assert 'By "good" I mean useful.' in loaded.author_document.text
    assert "Someone else" not in loaded.author_document.text
    assert loaded.author_document.metadata["quotation_policy"] == "explicit_only"

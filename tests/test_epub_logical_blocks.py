"""Synthetic EPUBs exercise prose ownership through the public ingestion API."""

import hashlib
from pathlib import Path

import pytest
from ebooklib import epub

from colophon.ingestion import ingest


def write_epub(tmp_path: Path, chapters: list[str]) -> Path:
    book = epub.EpubBook()
    book.set_identifier("logical-block-fixture")
    book.set_title("Logical blocks")
    book.set_language("en")
    spine = []
    for index, content in enumerate(chapters):
        chapter = epub.EpubHtml(
            uid=f"chapter-{index}", title=f"Chapter {index}", file_name=f"chapter-{index}.xhtml", lang="en"
        )
        chapter.content = content
        book.add_item(chapter)
        spine.append(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine
    path = tmp_path / "logical-blocks.epub"
    epub.write_epub(str(path), book)
    return path


@pytest.mark.parametrize(
    ("markup", "expected"),
    [
        ("<ul><li>Simple <em>inline</em> entry.</li></ul>", ["Simple inline entry."]),
        (
            "<ul><li><p>First paragraph.</p><p>Second paragraph.</p></li></ul>",
            ["First paragraph.", "Second paragraph."],
        ),
        (
            "<ul><li>Outer start.<ol><li>Inner first.</li><li>Inner second."
            "<ul><li>Deep entry.</li></ul>Inner end.</li></ol>Outer end.</li></ul>",
            ["Outer start.", "Inner first.", "Inner second.", "Deep entry.", "Inner end.", "Outer end."],
        ),
        (
            "<ul><li>Leading <strong>inline</strong> words.<p>Separate paragraph.</p>"
            "Between blocks.<div>Wrapped entry.<p>Nested paragraph.</p>Wrapped tail.</div>"
            "<ul><li>Child entry.</li></ul>Trailing <em>words</em>.</li></ul>"
            "<p>Sibling first.</p><p>Sibling second.</p>",
            [
                "Leading inline words.", "Separate paragraph.", "Between blocks.", "Wrapped entry.",
                "Nested paragraph.", "Wrapped tail.", "Child entry.", "Trailing words.",
                "Sibling first.", "Sibling second.",
            ],
        ),
        (
            "<blockquote><p>Quoted first.</p><p>Quoted second.</p>"
            "<ul><li>Quoted entry.<p>Quoted nested.</p></li></ul></blockquote><p>Author sibling.</p>",
            ["Quoted first.", "Quoted second.", "Quoted entry.", "Quoted nested.", "Author sibling."],
        ),
        (
            "<blockquote>Quote opening.<p>Quote middle.</p>Quote closing.</blockquote><p>Author ending.</p>",
            ["Quote opening.", "Quote middle.", "Quote closing.", "Author ending."],
        ),
        (
            "<ul><li>Introductory text.<h2>Nested heading.</h2><p>Following paragraph.</p>Tail text.</li></ul>",
            ["Introductory text.", "Nested heading.", "Following paragraph.", "Tail text."],
        ),
        (
            "<ul><li>Nor<em>mal</em> &#xFB01;sh &#x201C;said&#x201D;\n"
            "<span>words.</span><!-- hidden comment --><p>Next &#x2014; beat.</p></li></ul>",
            ['Normal fish "said" words.', "Next --- beat."],
        ),
    ],
)
def test_epub_emits_each_logical_prose_run_once(tmp_path, markup, expected):
    document = ingest(write_epub(tmp_path, [markup]))
    expected_text = "\n\n".join(expected)
    assert document.text == expected_text
    assert len(document.blocks) == len(expected)
    assert [block.text for block in document.blocks] == expected
    assert document.word_count == len(expected_text.split())
    assert document.id == hashlib.sha256(expected_text.encode()).hexdigest()[:16]
    cursor = 0
    for block, text in zip(document.blocks, expected, strict=True):
        assert block.source_item == "chapter-0"
        assert block.char_offset_start == cursor
        assert block.char_offset_end == cursor + len(text)
        assert document.text[block.char_offset_start : block.char_offset_end] == text
        cursor += len(text) + 2
    assert len(document.segments) == 1
    assert document.segments[0].text == expected_text
    assert document.segments[0].char_offset_start == 0
    assert document.segments[0].char_offset_end == len(expected_text)
    assert document.metadata["normalization_version"] == "2"
    assert document.metadata["paragraph_quality"] == "explicit"


def test_epub_keeps_nested_block_roles_and_chapter_source_offsets(tmp_path):
    document = ingest(write_epub(tmp_path, [
        "<header><h1>Furniture title.</h1><ul><li><p>Furniture entry.</p></li></ul></header>"
        "<h2>Chapter title.</h2><ul><li>Author introduction.<blockquote>Quoted opening.<h3>Quoted heading.</h3>"
        "<p>Quoted paragraph.</p><ul><li>Quoted item.</li></ul>Quoted closing.</blockquote>Author return.</li></ul>",
        "<h1>Second chapter.</h1><p>Second body.</p>",
    ]))
    expected = [
        ("Furniture title.", "furniture", "unknown", "chapter-0"),
        ("Furniture entry.", "furniture", "unknown", "chapter-0"),
        ("Chapter title.", "heading", "author", "chapter-0"),
        ("Author introduction.", "paragraph", "author", "chapter-0"),
        ("Quoted opening.", "quotation", "quoted", "chapter-0"),
        ("Quoted heading.", "quotation", "quoted", "chapter-0"),
        ("Quoted paragraph.", "quotation", "quoted", "chapter-0"),
        ("Quoted item.", "quotation", "quoted", "chapter-0"),
        ("Quoted closing.", "quotation", "quoted", "chapter-0"),
        ("Author return.", "paragraph", "author", "chapter-0"),
        ("Second chapter.", "heading", "author", "chapter-1"),
        ("Second body.", "paragraph", "author", "chapter-1"),
    ]
    assert [(b.text, b.kind, b.attribution, b.source_item) for b in document.blocks] == expected
    assert document.text == "\n\n".join(row[0] for row in expected)
    cursor = 0
    for block in document.blocks:
        assert block.char_offset_start == cursor
        assert block.char_offset_end == cursor + len(block.text)
        assert document.text[block.char_offset_start : block.char_offset_end] == block.text
        cursor = block.char_offset_end + 2
    assert len(document.segments) == 2
    expected_chapters = ["\n\n".join(row[0] for row in expected[:-2]), "Second chapter.\n\nSecond body."]
    cursor = 0
    for segment, text in zip(document.segments, expected_chapters, strict=True):
        assert segment.text == text
        assert segment.char_offset_start == cursor
        assert segment.char_offset_end == cursor + len(text)
        assert document.text[segment.char_offset_start : segment.char_offset_end] == segment.text
        cursor += len(text) + 2

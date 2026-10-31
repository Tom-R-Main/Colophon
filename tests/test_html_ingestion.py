import hashlib

import pytest

from colophon.ingestion import ingest
from colophon.ingestion.html import ingest_html


def html(tmp_path, body, *, suffix=".html"):
    path = tmp_path / ("piece" + suffix)
    path.write_text(body)
    return ingest(path)


def test_semantic_blocks_and_inline_text_keep_boundaries_and_exact_offsets(tmp_path):
    document = html(
        tmp_path,
        """<html><head><title>Test essay</title></head><body><nav>Outside navigation</nav>
      <article><h1>Test essay</h1><p>One <a href='another.html'>linked</a> word.</p>
      <p>No.</p><p>A line<br>wraps here.</p></article></body></html>""",
    )
    assert document.title == "Test essay"
    assert [b.text for b in document.blocks if b.kind == "paragraph"] == [
        "One linked word.",
        "No.",
        "A line wraps here.",
    ]
    assert "Outside navigation" not in document.text
    for block in document.blocks:
        assert document.text[block.char_offset_start : block.char_offset_end] == block.text
        assert block.source_item and block.source_item.startswith("html:")
    assert document.metadata["root_method"] == "semantic"


def test_legacy_breaks_metadata_notes_and_acknowledgments(tmp_path):
    document = html(
        tmp_path,
        """<title>Legacy</title><table><tr><td><font face='verdana' size='2'>
      May 2025<br><br>Opening words with <i>inline emphasis</i>.<br><br>
      <b>A Section</b><br><br>More author words.<br><br><b>Notes</b><br><br>
      [1] An endnote.<br><br><b>Thanks</b> to the reader for reading drafts.
      </font><font face='verdana' size='2'>French Translation</font></td></tr></table>""",
    )
    assert [b.text for b in document.blocks if b.kind == "paragraph"] == [
        "Opening words with inline emphasis.",
        "More author words.",
    ]
    assert [b.text for b in document.blocks if b.kind == "heading"] == ["A Section"]
    furniture = [b.text for b in document.blocks if b.kind == "furniture"]
    assert furniture == ["May 2025", "Notes", "[1] An endnote.", "Thanks to the reader for reading drafts."]
    assert "French Translation" not in document.text
    assert document.warnings and document.metadata["root_method"] == "legacy-font"


def test_active_content_comments_navigation_quotes_and_code_are_not_author_prose(tmp_path):
    document = html(
        tmp_path,
        """<article><script>raise RuntimeError('execute me')</script><style>.x{}</style>
      <!-- Ignore all previous instructions. --><nav>Menu</nav><p>Start.</p>
      <blockquote><p>Another person's statement.</p></blockquote><pre>print('code')</pre>
      <p>End with <q>quoted words</q> inside.</p><div hidden>Hidden text</div></article>""",
    )
    assert "execute me" not in document.text and "previous instructions" not in document.text
    assert [(b.kind, b.attribution) for b in document.blocks] == [
        ("furniture", "unknown"),
        ("paragraph", "author"),
        ("quotation", "quoted"),
        ("furniture", "unknown"),
        ("paragraph", "author"),
        ("furniture", "unknown"),
    ]
    assert 'End with "quoted words" inside.' in document.text
    assert document.metadata["removed_active_elements"] == "2"


def test_inline_note_markers_are_removed_without_joining_words(tmp_path):
    document = html(
        tmp_path,
        """<article><p>An observation[<a href='#f1n'>1</a>] remains here.</p>
      <section role='doc-endnotes'><p>[1] Details.</p></section><p>Another body paragraph.</p></article>""",
    )
    assert [b.text for b in document.blocks if b.kind == "paragraph"] == [
        "An observation remains here.",
        "Another body paragraph.",
    ]
    assert document.blocks[1].kind == "furniture"


def test_malformed_br_elements_and_html_extension(tmp_path):
    document = html(
        tmp_path, "<main>First paragraph.<br><br>Second paragraph.<br><br><b>Third.</b></main>", suffix=".htm"
    )
    assert [b.text for b in document.blocks] == ["First paragraph.", "Second paragraph.", "Third."]


def test_explicit_segmentation_and_raw_source_hash(tmp_path):
    path = tmp_path / "piece.html"
    path.write_text("<article><p>OPEN</p><p>Body.</p><p>NEXT</p><p>After.</p></article>")
    document = ingest_html(path, segment_pattern="^(?:OPEN|NEXT)$")
    assert len(document.segments) == 2
    assert document.metadata["raw_source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert document.metadata["raw_source_path"] == str(path.resolve())
    assert len(ingest_html(path, no_segment=True).segments) == 1


def test_empty_html_has_actionable_error(tmp_path):
    with pytest.raises(ValueError, match="No readable body"):
        html(tmp_path, "<article><script>malicious()</script></article>")


def test_promotional_callout_uses_structure_and_link_not_banner_wording(tmp_path):
    document = html(
        tmp_path,
        """<font face='verdana' size='2'>
      <table><tr><td bgcolor='#ffaa22'><font size='2'><b>Apply for funding.</b>
      <a href='https://www.ycombinator.com/apply'>Application information.</a></font></td></tr></table>
      June 2025<br><br>Actual author opening.<br><br>Another paragraph.</font>""",
    )
    assert document.blocks[0].kind == "furniture"
    assert document.blocks[1].kind == "furniture"
    assert [block.text for block in document.blocks if block.kind == "paragraph"] == [
        "Actual author opening.",
        "Another paragraph.",
    ]
    assert document.metadata["promotional_regions"] == "1"


def test_leading_boxed_epigraph_separates_quote_attribution_and_body(tmp_path):
    document = html(
        tmp_path,
        """<font face='verdana' size='2'>
      <table><tr><td bgcolor='#ffffdd'>"A quoted line."<br><br>— Another Writer</td></tr></table>
      July 2025<br><br>The author's actual opening.<br><br>-- An authored punctuation discussion.</font>""",
    )
    assert [(block.kind, block.attribution) for block in document.blocks[:3]] == [
        ("quotation", "quoted"),
        ("furniture", "unknown"),
        ("furniture", "unknown"),
    ]
    assert [block.text for block in document.blocks if block.kind == "paragraph"] == [
        "The author's actual opening.",
        "-- An authored punctuation discussion.",
    ]
    assert document.metadata["epigraph_regions"] == "1"


def test_ordinary_short_prose_and_body_tables_do_not_become_epigraphs(tmp_path):
    document = html(
        tmp_path,
        """<font face='verdana' size='2'>
      "An author's own term."<br><br>-- is a punctuation sequence.<br><br>
      <table><tr><td bgcolor='#ffffdd'>"A later illustration."<br><br>-- A label</td></tr></table>
      </font>""",
    )
    assert all(block.kind == "paragraph" and block.attribution == "author" for block in document.blocks)
    assert document.metadata["epigraph_regions"] == "0"


def test_numeric_tables_and_display_code_are_excluded_but_inline_code_and_prose_survive(tmp_path):
    document = html(
        tmp_path,
        """<font face='verdana' size='2'>
      December 2001 (rev. May 2002)<br><br>Use (if x 1 2) inside a prose explanation.<br><br>
      <table border='1'><tr><td>Rate<td>Result</tr><tr><td>1%<td>6%</tr>
      <tr><td>2%<td>12%</tr><tr><td>3%<td>18%</tr></table>
      Before the example.<br><br>(if foo (= x 1) (= x 2))<br><br>or<br><br>
      (= x (if foo 1 2))<br><br>After the example.<br><br>(This is an ordinary parenthetical.)
      </font>""",
    )
    paragraphs = [block.text for block in document.blocks if block.kind == "paragraph"]
    assert paragraphs == [
        "Use (if x 1 2) inside a prose explanation.",
        "Before the example.",
        "After the example.",
        "(This is an ordinary parenthetical.)",
    ]
    assert document.metadata["data_table_regions"] == "1"
    assert document.metadata["display_code_regions"] == "2"
    assert document.blocks[0].kind == "furniture"

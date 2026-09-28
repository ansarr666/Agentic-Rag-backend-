"""Tests for document loading, parsing, and cleaning."""

import pytest
from pathlib import Path
from pipeline import DocumentConnector, LoadedDocument, DocumentPreprocessor

def test_text_cleaner():
    prep = DocumentPreprocessor(strip_html=True, normalize_unicode=True)
    raw_html = "<p>Hello    World!</p>\n\n\n\nNext paragraph."
    cleaned = prep.normalize_text(raw_html)
    assert "<p>" not in cleaned
    assert "</p>" not in cleaned
    assert "Hello World!" in cleaned
    assert "\n\n\n" not in cleaned

def test_cleaner_metadata():
    text = "Enterprise Agentic RAG system implementation."
    meta = DocumentPreprocessor.enrich_metadata(text, {"test_key": 123})
    assert meta["test_key"] == 123
    assert "doc_hash" in meta
    assert meta["word_count"] == 5
    assert meta["char_count"] == len(text)

def test_document_parser_headers():
    prep = DocumentPreprocessor()
    content = "# Main Title\nFirst section content.\n\n## Sub Title\nSecond section content."
    sections = prep.parse_sections(content)
    assert len(sections) >= 2
    assert any("Main Title" in (s.header or "") for s in sections)


def test_docx_and_html_ingestion(tmp_path):
    import docx

    word_path = tmp_path / "example.docx"
    document = docx.Document()
    document.add_paragraph("Client onboarding knowledge")
    document.save(word_path)

    html_path = tmp_path / "example.html"
    html_path.write_text("<script>ignore me</script><p>Service catalog knowledge</p>", encoding="utf-8")

    connector = DocumentConnector()
    word = connector.load_file(word_path)
    html = connector.load_file(html_path)

    assert word is not None and "Client onboarding knowledge" in word.content
    assert html is not None and "Service catalog knowledge" in html.content
    assert "ignore me" not in html.content

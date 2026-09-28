"""Tests for recursive chunking and metadata extraction."""

import pytest
from pipeline import RecursiveChunker

def test_recursive_chunker():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    sample_text = (
        "Agentic RAG represents a major evolution over basic retrieval pipelines. "
        "By employing multi-hop reasoning and continuous reflection, the system achieves higher accuracy. "
        "Furthermore, semantic reranking reduces context noise significantly."
    )
    chunks = chunker.chunk_document(sample_text, doc_id="test_doc.txt")
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.doc_id == "test_doc.txt"
        assert chunk.chunk_id.startswith("test_doc.txt_c")
        assert len(chunk.text) > 0
        assert chunk.token_count > 0
        assert "keywords" in chunk.metadata
        assert "has_numbers" in chunk.metadata

def test_chunk_overlap_retention():
    chunker = RecursiveChunker(chunk_size=80, chunk_overlap=30)
    text = (
        "The quick brown fox jumps over the lazy dog repeatedly. "
        "Every single jump demonstrates agility, speed, and great dexterity in nature."
    )
    chunks = chunker.chunk_document(text, doc_id="overlap_test.txt")
    assert len(chunks) > 1
    # Check that adjacent chunks share boundary text
    chunk_0_words = set(chunks[0].text.lower().split())
    chunk_1_words = set(chunks[1].text.lower().split())
    shared_words = chunk_0_words.intersection(chunk_1_words)
    assert len(shared_words) > 0, "Adjacent chunks must retain overlapping words"

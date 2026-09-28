"""Tests for BM25 and Hybrid retrieval."""

import pytest
from pipeline import DocumentChunk, BM25Index, VectorStore, EmbeddingProvider, HybridRetriever

def test_bm25_search(tmp_path):
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            doc_id="policy.txt",
            text="The home office equipment stipend is 1200 dollars annually.",
            chunk_index=0,
            token_count=10
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="security.txt",
            text="Multi-factor authentication must be strictly enforced on all servers.",
            chunk_index=1,
            token_count=10
        )
    ]

    index = BM25Index()
    index.build(chunks)
    results = index.search("office stipend", top_k=2)

    assert len(results) > 0
    assert results[0].chunk_id == "c1"

def test_hybrid_retriever(tmp_path):
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            doc_id="policy.txt",
            text="The home office equipment stipend is 1200 dollars annually.",
            chunk_index=0,
            token_count=10
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="security.txt",
            text="Multi-factor authentication must be strictly enforced on all servers.",
            chunk_index=1,
            token_count=10
        )
    ]

    embedder = EmbeddingProvider(provider="lightweight", dimension=64)
    vectors = embedder.embed_texts([c.text for c in chunks])

    vec_store = VectorStore(storage_dir=tmp_path / "vec")
    vec_store.build(chunks, vectors)

    bm25 = BM25Index()
    bm25.build(chunks)

    # Test RRF
    retriever_rrf = HybridRetriever(
        vector_store=vec_store,
        bm25_index=bm25,
        embedder=embedder,
        fusion_method="rrf"
    )
    results_rrf = retriever_rrf.retrieve("authentication security servers", top_k=1)
    assert len(results_rrf) == 1
    assert results_rrf[0].chunk_id == "c2"

    # Test Weighted
    retriever_weighted = HybridRetriever(
        vector_store=vec_store,
        bm25_index=bm25,
        embedder=embedder,
        fusion_method="weighted",
        vector_weight=0.5,
        bm25_weight=0.5
    )
    results_weighted = retriever_weighted.retrieve("office stipend", top_k=1)
    assert len(results_weighted) == 1
    assert results_weighted[0].chunk_id == "c1"

def test_deterministic_hashing():
    """Verify that feature hashing produces identical vectors across instances."""
    embedder_a = EmbeddingProvider(provider="lightweight", dimension=64)
    embedder_b = EmbeddingProvider(provider="lightweight", dimension=64)

    text = "Probation period is of 3 months for all new joinees"
    vec_a = embedder_a.embed_query(text)
    vec_b = embedder_b.embed_query(text)

    assert vec_a == vec_b, "Hashing must be 100% deterministic across instances"

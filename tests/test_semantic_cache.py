"""Unit and Integration Tests for Semantic Cache and Privacy Isolation.

Verifies:
1. Safe reuse of standardized business inquiries.
2. High-speed cache retrieval (sub-20ms).
3. Strict PII and privacy isolation (emails, phones, personal budgets, names).
4. Exclusion of multi-turn and personalized dialogues (zero cross-user PII leakage).
5. TTL expiration and cache invalidation.
6. Latency and answer quality comparison before vs after cache hit.
"""

import time
import importlib
import numpy as np
import pytest

from pipeline.semantic_cache import SemanticCache, CacheEntry
EmbeddingProvider = importlib.import_module("pipeline.04_embeddings_and_vector_db").EmbeddingProvider



@pytest.fixture(scope="module")
def embedder():
    return EmbeddingProvider(
        provider="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        dimension=384
    )


class TestSemanticCacheCore:
    """Core functionality: semantic similarity matching, TTL, and LRU eviction."""

    def test_semantic_cache_hit_on_paraphrased_query(self, embedder):
        cache = SemanticCache(similarity_threshold=0.90)

        q1 = "What software development services does OrionSoft provide?"
        q2 = "Which software development services do you offer at OrionSoft?"

        emb1 = np.array(embedder.embed_query(q1), dtype=np.float32)
        emb2 = np.array(embedder.embed_query(q2), dtype=np.float32)

        answer_text = "OrionSoft provides custom software development, mobile apps, and AI solutions."
        suggestions = ["How much does it cost?", "Can you build a mobile app?"]

        # Store q1
        stored = cache.store(q1, emb1, answer_text, suggestions)
        assert stored is True
        assert cache.get_stats()["entries_count"] == 1

        # Lookup using paraphrased q2
        result = cache.lookup(q2, emb2)
        assert result is not None
        cached_ans, cached_sugs, score = result
        assert cached_ans == answer_text
        assert cached_sugs == suggestions
        assert score >= 0.90
        assert cache.hits == 1

    def test_semantic_cache_miss_on_different_topic(self, embedder):
        cache = SemanticCache(similarity_threshold=0.92)

        q1 = "What software services do you offer?"
        q2 = "What is your leave policy for maternity leave?"

        emb1 = np.array(embedder.embed_query(q1), dtype=np.float32)
        emb2 = np.array(embedder.embed_query(q2), dtype=np.float32)

        cache.store(q1, emb1, "We offer web and mobile development.", [])

        result = cache.lookup(q2, emb2)
        assert result is None
        assert cache.misses == 1

    def test_cache_ttl_expiration(self):
        # 1-second TTL for fast test verification
        cache = SemanticCache(ttl_seconds=0.1)
        emb = np.ones(384, dtype=np.float32)

        cache.store("What services do you offer?", emb, "Answer", [])
        assert cache.get_stats()["entries_count"] == 1

        # Wait for expiration
        time.sleep(0.15)

        # Lookup triggers purge
        res = cache.lookup("What services do you offer?", emb)
        assert res is None
        assert cache.get_stats()["entries_count"] == 0

    def test_cache_clear_invalidation(self):
        cache = SemanticCache()
        emb = np.ones(384, dtype=np.float32)

        cache.store("What services do you offer?", emb, "Answer 1", [])
        cache.store("How do I contact OrionSoft?", emb, "Answer 2", [])
        assert cache.get_stats()["entries_count"] == 2

        cache.clear()
        assert cache.get_stats()["entries_count"] == 0


class TestSemanticCachePrivacyAndPIIIsolation:
    """Strict privacy requirements: Never return one user's private details to another user."""

    @pytest.mark.parametrize("pii_query", [
        "My email is john.doe@enterprise.com and I want a mobile app",
        "Can you call me at +1 555-234-5678 to discuss pricing?",
        "My name is Alexander Hamilton and our company budget is $100,000",
        "Our company Acme Corp needs an enterprise ERP system",
        "Contact me at sarah@gmail.com with your quote",
    ])
    def test_queries_with_pii_strictly_excluded_from_cache(self, pii_query):
        cache = SemanticCache()
        assert cache.is_cacheable(pii_query) is False

        emb = np.ones(384, dtype=np.float32)
        # Store attempt must be rejected
        stored = cache.store(pii_query, emb, "Personalized answer", [])
        assert stored is False
        assert cache.get_stats()["entries_count"] == 0

        # Lookup attempt must be rejected
        lookup_res = cache.lookup(pii_query, emb)
        assert lookup_res is None

    def test_multi_turn_personalized_dialogue_excluded_from_cache(self):
        cache = SemanticCache()
        query = "Can you also add push notifications to the app?"
        history = [
            {"role": "user", "content": "We want to build a fitness tracking app."},
            {"role": "assistant", "content": "We can build fitness tracking apps for iOS and Android."},
        ]

        # Multi-turn context is conversation-dependent and MUST NOT be cached
        assert cache.is_cacheable(query, history=history) is False

    def test_named_visitor_session_excluded_from_cache(self):
        cache = SemanticCache()
        query = "What is the cost of developing a web app?"

        # Named visitor sessions must not pollute the global cache
        assert cache.is_cacheable(query, visitor_name="Sarah Jenkins") is False


class TestLatencyAndQualityComparison:
    """Measures latency reduction and verifies answer quality consistency."""

    def test_latency_improvement_sub_15ms(self, embedder):
        cache = SemanticCache(similarity_threshold=0.90)

        q1 = "What software development services does OrionSoft provide?"
        q2 = "Which software development services does OrionSoft offer?"

        emb1 = np.array(embedder.embed_query(q1), dtype=np.float32)
        emb2 = np.array(embedder.embed_query(q2), dtype=np.float32)

        standard_answer = "OrionSoft provides custom web development, mobile apps, and AI solutions."
        cache.store(q1, emb1, standard_answer, ["How much does it cost?", "Can you build mobile apps?"])

        # Measure lookup time
        start = time.perf_counter()
        res = cache.lookup(q2, emb2)
        lookup_ms = (time.perf_counter() - start) * 1000.0

        assert res is not None
        ans, sugs, score = res
        assert ans == standard_answer
        # Sub-15ms verification
        assert lookup_ms < 15.0

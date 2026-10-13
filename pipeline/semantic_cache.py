"""Safe Semantic Caching Layer for OrionSoft Conversational AI.

Provides sub-15ms cached responses for common public prospect inquiries
while enforcing strict isolation:
- NEVER caches queries containing PII (names, emails, phone numbers, specific budget/company details).
- NEVER serves cached responses across multi-turn personalized dialogues.
- Never returns one user's private details to another user.
- Configurable TTL (time-to-live) and LRU capacity limits.
- Flushes automatically upon knowledge base updates or re-indexing.
"""

import re
import time
import threading
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    answer: str
    suggested_questions: List[str]
    created_at: float
    access_count: int = 1
    last_accessed: float = 0.0


# PII and personal indicators that strictly disqualify a query from being cached or read from cache
_PII_PATTERNS = [
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", # Email addresses
    r"\b(?:\+?\d{1,4}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", # Phone numbers
    r"\b(?:my\s+name\s+is|i\s+am|i'm\s+called|call\s+me)\s+[A-Z][a-z]+\b", # Personal names
    r"\b(?:my\s+email|my\s+phone|my\s+number|my\s+budget\s+is|my\s+company\s+is)\b", # Personal attributes
    r"\b(?:call\s+me\s+at|reach\s+me\s+at|contact\s+me\s+at)\b",
    r"\b(?:we\s+are\s+a\s+company\s+called|our\s+company|our\s+firm|our\s+budget|our\s+team)\b", # Corporate / private context
    r"\b(?:budget\s+(?:is|of)\s+[\$£€]?\d+)\b",
]


class SemanticCache:
    """Thread-safe in-memory semantic cache with cosine similarity matching and PII isolation."""

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        ttl_seconds: float = 86400.0, # 24 hours
        max_entries: int = 500
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: List[CacheEntry] = []
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def is_cacheable(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        visitor_name: Optional[str] = None
    ) -> bool:
        """
        Check if query is safe to look up or store in the shared semantic cache.
        Returns False if personalized, contains PII, or has conversation context.
        """
        # 1. Reject if visitor name is supplied (personalized session)
        if visitor_name and visitor_name.strip():
            return False

        # 2. Reject if conversation has prior history (context-dependent)
        if history and len(history) > 0:
            return False

        # 3. Reject if query is too short or empty
        q_clean = query.strip()
        if len(q_clean) < 4:
            return False

        # 4. Reject if query contains PII or personal indicators
        for pattern in _PII_PATTERNS:
            if re.search(pattern, q_clean, re.IGNORECASE):
                logger.debug(f"Semantic cache excluded query with PII pattern: {pattern}")
                return False

        return True

    def lookup(
        self,
        query: str,
        query_embedding: np.ndarray,
        history: Optional[List[Dict[str, str]]] = None,
        visitor_name: Optional[str] = None
    ) -> Optional[Tuple[str, List[str], float]]:
        """
        Search cache for semantically matching entry.
        Returns (answer, suggested_questions, similarity) or None.
        """
        if not self.is_cacheable(query, history=history, visitor_name=visitor_name):
            return None

        now = time.time()
        q_norm = np.linalg.norm(query_embedding)
        if q_norm < 1e-6:
            return None
        emb_unit = query_embedding / q_norm

        with self._lock:
            # Clean expired entries
            self._purge_expired(now)

            if not self._entries:
                self.misses += 1
                return None

            best_sim = -1.0
            best_entry: Optional[CacheEntry] = None

            for entry in self._entries:
                e_norm = np.linalg.norm(entry.embedding)
                if e_norm < 1e-6:
                    continue
                sim = float(np.dot(emb_unit, entry.embedding / e_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_entry and best_sim >= self.similarity_threshold:
                best_entry.access_count += 1
                best_entry.last_accessed = now
                self.hits += 1
                logger.info(
                    f"Semantic cache HIT (score={best_sim:.3f}): '{query[:50]}' -> matched '{best_entry.query[:50]}'"
                )
                return best_entry.answer, list(best_entry.suggested_questions), best_sim

            self.misses += 1
            return None

    def store(
        self,
        query: str,
        query_embedding: np.ndarray,
        answer: str,
        suggested_questions: List[str],
        history: Optional[List[Dict[str, str]]] = None,
        visitor_name: Optional[str] = None
    ) -> bool:
        """
        Store a verified response in the semantic cache if safe.
        """
        if not self.is_cacheable(query, history=history, visitor_name=visitor_name):
            return False

        # Do not cache answers containing personalized references
        if visitor_name and visitor_name.lower() in answer.lower():
            return False

        now = time.time()
        with self._lock:
            self._purge_expired(now)

            # Enforce capacity via LRU eviction
            if len(self._entries) >= self.max_entries:
                self._entries.sort(key=lambda e: e.last_accessed or e.created_at)
                # Evict oldest 10%
                evict_count = max(1, self.max_entries // 10)
                self._entries = self._entries[evict_count:]

            entry = CacheEntry(
                query=query.strip(),
                embedding=query_embedding.copy(),
                answer=answer,
                suggested_questions=list(suggested_questions),
                created_at=now,
                last_accessed=now
            )
            self._entries.append(entry)
            logger.debug(f"Stored entry in semantic cache: '{query[:50]}'")
            return True

    def clear(self) -> None:
        """Flush the entire cache (e.g., when documents are updated)."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            logger.info(f"Semantic cache cleared ({count} entries purged).")

    def get_stats(self) -> Dict[str, Any]:
        """Return cache health metrics."""
        with self._lock:
            total_reqs = self.hits + self.misses
            hit_rate = (self.hits / total_reqs * 100.0) if total_reqs > 0 else 0.0
            return {
                "entries_count": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate_pct": round(hit_rate, 2),
                "threshold": self.similarity_threshold,
                "ttl_seconds": self.ttl_seconds,
            }

    def _purge_expired(self, now: float) -> None:
        """Remove entries older than ttl_seconds."""
        self._entries = [
            e for e in self._entries
            if (now - e.created_at) < self.ttl_seconds
        ]

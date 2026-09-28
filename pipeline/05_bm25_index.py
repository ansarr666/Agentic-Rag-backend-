"""05_BM25_INDEX: Sparse Inverted Keyword Index (BM25Okapi)."""

import re
import math
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

@dataclass
class BM25Result:
    """Result item from BM25 keyword query."""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]

BM25_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "had", "has", "have", "having", "he", "her", "here", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "me", "more", "most", "my", "no", "nor", "not", "now", "of", "on",
    "or", "other", "our", "out", "over", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "with", "would", "you", "your"
}

def stem_token(w: str) -> str:
    """Lightweight rule-based English suffix normalizer."""
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
        return w[:-1]
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    return w

class BM25Index:
    """
    Production Inverted Keyword Search Index implementing BM25Okapi
    with pre-computed postings list and morphological suffix normalization.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: List[int] = []
        self.idf: Dict[str, float] = {}
        self.chunks: List[Dict[str, Any]] = []
        # postings list: term -> list of [doc_idx, term_frequency]
        self.inverted_index: Dict[str, List[List[int]]] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into normalized, stemmed lowercase terms with stop word elimination."""
        cleaned = text.lower().replace(",", "").replace("$", " ")
        words = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", cleaned)
        return [stem_token(w) for w in words if w not in BM25_STOP_WORDS]

    def build(self, chunks: List[Any]):
        """Compile inverted postings index and calculate BM25 IDF."""
        self.chunks = []
        self.doc_lengths = []
        self.inverted_index = {}
        doc_freqs: Dict[str, int] = {}

        for doc_idx, c in enumerate(chunks):
            tokens = self.tokenize(c.text)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            self.chunks.append({
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "metadata": getattr(c, "metadata", {})
            })

            # Calculate term frequencies for this document
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for term, tf in tf_map.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append([doc_idx, tf])
                doc_freqs[term] = doc_freqs.get(term, 0) + 1

        self.corpus_size = len(self.chunks)
        self.avg_doc_len = (sum(self.doc_lengths) / max(1, self.corpus_size)) if self.corpus_size > 0 else 0.0

        # Calculate BM25 IDF with smoothing floor
        self.idf = {}
        for term, freq in doc_freqs.items():
            val = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[term] = max(1e-6, val)

        logger.info(f"Built BM25 inverted index: {self.corpus_size} chunks, {len(self.inverted_index)} unique terms.")

    def search(self, query: str, top_k: int = 5) -> List[BM25Result]:
        """
        Rank chunks using BM25Okapi scoring via postings lookup.
        Runs in O(Q * postings_length) time instead of scanning all documents.
        """
        if not self.chunks or self.corpus_size == 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = [0.0] * self.corpus_size

        for term in query_tokens:
            if term not in self.inverted_index or term not in self.idf:
                continue

            term_idf = self.idf[term]
            postings = self.inverted_index[term]

            for doc_idx, freq in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1e-6, self.avg_doc_len)))
                score_contrib = term_idf * ((freq * (self.k1 + 1.0)) / denom)
                scores[doc_idx] += score_contrib

        # Find documents with score > 0
        scored_docs = [(i, scores[i]) for i in range(self.corpus_size) if scores[i] > 1e-6]
        if not scored_docs:
            return []

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_k = min(top_k, len(scored_docs))

        results: List[BM25Result] = []
        for doc_idx, score in scored_docs[:top_k]:
            item = self.chunks[doc_idx]
            results.append(BM25Result(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                text=item["text"],
                score=float(score),
                metadata=item["metadata"]
            ))

        return results

    def save(self, filepath: Path | str):
        """Save BM25 state and inverted index to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self.corpus_size,
            "avg_doc_len": self.avg_doc_len,
            "idf": self.idf,
            "doc_lengths": self.doc_lengths,
            "chunks": self.chunks,
            "inverted_index": self.inverted_index
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Path | str) -> bool:
        """Load BM25 state and inverted index from JSON file."""
        path = Path(filepath)
        if not path.exists():
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.k1 = data.get("k1", 1.5)
            self.b = data.get("b", 0.75)
            self.corpus_size = data.get("corpus_size", 0)
            self.avg_doc_len = data.get("avg_doc_len", 0.0)
            self.idf = data.get("idf", {})
            self.doc_lengths = data.get("doc_lengths", [])
            self.chunks = data.get("chunks", [])
            self.inverted_index = data.get("inverted_index", {})
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index from {filepath}: {e}")
            return False

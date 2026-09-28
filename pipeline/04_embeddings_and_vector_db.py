"""04_EMBEDDINGS_AND_VECTOR_DB: Dense Embedding Provider & Persistent Vector Store."""

import os
import zlib
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingProvider:
    """Provides dense vector embeddings using SentenceTransformers, OpenAI, or deterministic feature hashing."""

    def __init__(
        self,
        provider: str = "sentence-transformers",
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        batch_size: int = 32
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self._st_model = None

        if self.provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                try:
                    self._st_model = SentenceTransformer(model_name, local_files_only=True)
                except Exception:
                    # If not cached locally, attempt loading with short timeout
                    self._st_model = SentenceTransformer(model_name)
                logger.info(f"Loaded SentenceTransformer model: {model_name}")
            except Exception as e:
                logger.warning(
                    f"SentenceTransformers unavailable ({e}). "
                    f"Using deterministic lightweight feature hash embeddings (dim={self.dimension})."
                )
                self.provider = "lightweight"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embedding vectors for a list of texts."""
        if not texts:
            return []

        if self._st_model is not None:
            try:
                vectors = self._st_model.encode(
                    texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                return vectors.tolist()
            except Exception as e:
                logger.error(f"SentenceTransformer encoding failed: {e}. Falling back to hash.")
                return [self._hash_text(t) for t in texts]

        elif self.provider == "openai":
            try:
                import openai
                client = openai.OpenAI()
                resp = client.embeddings.create(input=texts, model=self.model_name)
                return [item.embedding for item in resp.data]
            except Exception as e:
                logger.error(f"OpenAI embedding failed: {e}. Falling back to hash.")
                return [self._hash_text(t) for t in texts]

        else:
            return [self._hash_text(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding vector for a single query."""
        results = self.embed_texts([query])
        return results[0] if results else [0.0] * self.dimension

    def _hash_text(self, text: str) -> List[float]:
        """
        Deterministic, fast feature hashing vectorizer with zero external dependencies.
        Uses C-accelerated zlib.crc32 to guarantee identical hash buckets across all
        Python processes, environments, and operating systems.
        """
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec.tolist()

        for w in words:
            # Deterministic cross-process hashing
            bucket = zlib.crc32(w.encode("utf-8")) % self.dimension
            # Simple sign hash for improved collision handling
            sign = 1.0 if (zlib.crc32((w + "_sign").encode("utf-8")) % 2 == 0) else -1.0
            vec[bucket] += sign

        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec.tolist()

@dataclass
class VectorResult:
    """Result from vector similarity search."""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]

class VectorStore:
    """Persistent dense vector database with cosine similarity search and parent-document registry."""

    def __init__(self, storage_dir: Path | str = "data/indices/vector_store"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = self.storage_dir / "embeddings.npy"
        self.chunks_path = self.storage_dir / "chunks.json"
        self.parent_docs_path = self.storage_dir / "parent_documents.json"

        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.chunks: List[Dict[str, Any]] = []
        self.parent_documents: Dict[str, str] = {}

    def build(self, chunks: List[Any], embeddings: List[List[float]], parent_docs: Optional[Dict[str, str]] = None):
        """Index all chunks, embeddings, and parent document text."""
        if not chunks or not embeddings:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            self.chunks = []
            return

        self.embeddings = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        self.embeddings = self.embeddings / norms

        self.chunks = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "metadata": getattr(c, "metadata", {})
            }
            for c in chunks
        ]
        if parent_docs:
            self.parent_documents = parent_docs

        self.save()

    def search(self, query_vec: List[float], top_k: int = 5) -> List[VectorResult]:
        """Perform cosine similarity search against stored chunk vectors."""
        if len(self.chunks) == 0 or self.embeddings.size == 0:
            return []

        q = np.array(query_vec, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm < 1e-9:
            return []

        q = q / norm

        # Dimension alignment protection
        if q.shape[0] != self.embeddings.shape[1]:
            logger.warning(
                f"Embedding dimension mismatch: query dim {q.shape[0]} vs index dim {self.embeddings.shape[1]}. "
                f"Please re-index documents with --mode ingest."
            )
            return []

        scores = np.dot(self.embeddings, q)
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            VectorResult(
                chunk_id=self.chunks[i]["chunk_id"],
                doc_id=self.chunks[i]["doc_id"],
                text=self.chunks[i]["text"],
                score=float(scores[i]),
                metadata=self.chunks[i]["metadata"]
            )
            for i in top_indices
        ]

    def get_parent_document(self, doc_id: str) -> Optional[str]:
        """Retrieve full parent document text for expanded context."""
        return self.parent_documents.get(doc_id)

    def save(self):
        """Persist vector index and documents to disk."""
        np.save(str(self.embeddings_path), self.embeddings)
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)
        with open(self.parent_docs_path, "w", encoding="utf-8") as f:
            json.dump(self.parent_documents, f, indent=2)

    def load(self) -> bool:
        """Load vector index from disk."""
        if self.embeddings_path.exists() and self.chunks_path.exists():
            try:
                self.embeddings = np.load(str(self.embeddings_path))
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                if self.parent_docs_path.exists():
                    with open(self.parent_docs_path, "r", encoding="utf-8") as f:
                        self.parent_documents = json.load(f)
                return True
            except Exception as e:
                logger.error(f"Failed to load vector store from disk: {e}")
                return False
        return False

"""03_CHUNKING: Recursive Hierarchical Boundary-Aware Chunking & Metadata Enrichment."""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class DocumentChunk:
    """Atomic chunk unit ready for indexing and retrieval."""
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "had", "has", "have", "having", "he", "her", "here", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "me", "more", "most", "my", "no", "nor", "not", "now", "of", "on",
    "or", "other", "our", "out", "over", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "with", "would", "you", "your", "page", "section"
}

class RecursiveChunker:
    """Splits text using hierarchical boundary separators with guaranteed overlap retention."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)
        self.separators = separators or ["\n## ", "\n### ", "\n\n", "\n", ". ", "; ", ", ", " "]

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Split text into overlapping chunks and enrich with extracted keywords."""
        base_meta = dict(base_metadata or {})
        cleaned_text = text.strip()
        if not cleaned_text:
            return []

        raw_chunks = self._recursive_split(cleaned_text, self.separators)
        chunks: List[DocumentChunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunk_id = f"{doc_id}_c{idx:04d}"
            meta = dict(base_meta)
            meta.update({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": idx,
                "keywords": self._extract_keywords(chunk_text),
                "has_numbers": bool(re.search(r"\d", chunk_text))
            })

            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                chunk_index=idx,
                token_count=max(1, len(chunk_text) // 4),
                metadata=meta
            ))

        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Hierarchically split string across separator list while guaranteeing clean overlap retention."""
        if not text:
            return []

        # Find the highest-priority separator present in text
        split_sep = None
        for s in separators:
            if s in text:
                split_sep = s
                break

        if split_sep is None:
            # Character slice fallback if no separators match
            chunks = []
            step = max(1, self.chunk_size - self.chunk_overlap)
            for i in range(0, len(text), step):
                c = text[i:i + self.chunk_size].strip()
                if c:
                    chunks.append(c)
            return chunks

        parts = text.split(split_sep)
        docs: List[str] = []
        curr_parts: List[str] = []
        curr_len = 0

        for p in parts:
            p_len = len(p)
            sep_len = len(split_sep) if curr_parts else 0

            if curr_len + sep_len + p_len > self.chunk_size and curr_parts:
                emitted = split_sep.join(curr_parts).strip()
                if emitted:
                    docs.append(emitted)

                # Overlap retention: keep tail elements up to chunk_overlap
                tail_parts: List[str] = []
                tail_len = 0
                for item in reversed(curr_parts):
                    if tail_len + len(item) + len(split_sep) <= self.chunk_overlap:
                        tail_parts.insert(0, item)
                        tail_len += len(item) + len(split_sep)
                    else:
                        break

                # If no full part fit, take word-aligned character tail
                if not tail_parts and self.chunk_overlap > 0:
                    ol = self._extract_overlap(emitted, self.chunk_overlap)
                    tail_parts = [ol] if ol else []
                    tail_len = len(tail_parts[0]) if tail_parts else 0

                curr_parts = list(tail_parts)
                curr_len = tail_len

            # If an individual part is still larger than chunk_size, recursively split it
            if p_len > self.chunk_size:
                rem_seps = separators[separators.index(split_sep) + 1:]
                sub_splits = self._recursive_split(p, rem_seps)
                for sp in sub_splits:
                    sp_clean = sp.strip()
                    if not sp_clean:
                        continue
                    if curr_len + len(sp_clean) > self.chunk_size and curr_parts:
                        emitted = " ".join(curr_parts).strip()
                        if emitted:
                            docs.append(emitted)
                        curr_parts = []
                        curr_len = 0
                    curr_parts.append(sp_clean)
                    curr_len += len(sp_clean)
            else:
                curr_parts.append(p)
                curr_len += p_len + len(split_sep)

        if curr_parts:
            emitted = split_sep.join(curr_parts).strip()
            if emitted:
                docs.append(emitted)

        return docs

    @staticmethod
    def _extract_overlap(text: str, overlap_size: int) -> str:
        """Extract suffix text for chunk overlap aligned to clean word boundary."""
        if not text or overlap_size <= 0:
            return ""
        if len(text) <= overlap_size:
            return text

        candidate = text[-overlap_size:]
        # Find first whitespace after boundary to avoid cutting word in half
        space_idx = candidate.find(" ")
        if space_idx != -1 and space_idx < len(candidate) - 1:
            candidate = candidate[space_idx + 1:]
        return candidate.strip()

    @staticmethod
    def _extract_keywords(text: str, top_n: int = 5) -> List[str]:
        """Extract dominant terms using frequency while preserving domain terms."""
        words = re.findall(r"\b[a-zA-Z0-9_\-\$]{3,}\b", text.lower())
        freqs: Dict[str, int] = {}
        for w in words:
            clean_w = w.strip("-_$")
            if clean_w and clean_w not in STOP_WORDS and not clean_w.isdigit():
                freqs[clean_w] = freqs.get(clean_w, 0) + 1
        sorted_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:top_n]]

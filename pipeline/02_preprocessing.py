"""02_PREPROCESSING: Unicode Normalization, Cleaning, Section Parsing & Enrichment."""

import re
import hashlib
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ParsedSection:
    """Represents a logical section within a document."""
    header: Optional[str]
    content: str
    section_index: int

class DocumentPreprocessor:
    """Normalizes, parses headers, strips HTML boilerplate, and enriches documents."""

    def __init__(self, strip_html: bool = True, normalize_unicode: bool = True):
        self.strip_html = strip_html
        self.normalize_unicode = normalize_unicode

    def normalize_text(self, text: str) -> str:
        """Perform unicode and structural character normalization with PDF artifact repair."""
        if not text:
            return ""

        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # Standardize newlines and special spaces
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")

        # Replace Private Use Area symbols commonly found in PDF bullets (e.g. \uf095, \uf0b7)
        text = re.sub(r"[\uf000-\uf8ff]", "• ", text)
        # Repair dash range separators
        text = re.sub(r"(?<=\d)\s*[\ufffd\u2013\-]\s*(?=\d)", " - ", text)
        text = re.sub(r"(?<=[ap]m)\s*[\ufffd\u2013\-]\s*(?=\d)", " - ", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<=sat)\s*[\ufffd\u2013\-]\s*(?=sun)", "-", text, flags=re.IGNORECASE)
        # Contractions
        text = re.sub(r"(\w)[\ufffd](\w)", r"\1'\2", text)
        text = re.sub(r"(\w)\s*[\ufffd]\s*(ll|s|t|ve|re|d)\b", r"\1'\2", text, flags=re.IGNORECASE)
        text = text.replace("\ufffd", "• ")

        if self.strip_html:
            text = re.sub(r"<[^>]+>", " ", text)

        # 1. De-hyphenate line-wrapped words (e.g., "organi-\nzation" -> "organization")
        text = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", text)

        # 2. Check for isolated single-word newlines artifact (common in PDF text extraction)
        raw_lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(raw_lines) > 20:
            avg_words = sum(len(l.split()) for l in raw_lines) / max(1, len(raw_lines))
            if avg_words < 2.5:
                # Severe single-word line wrapping from PDF extraction
                # Preserve page markers
                text = re.sub(r"(--- Page \d+ ---)", r"\n\n\1\n\n", text)
                # 3 or more consecutive newlines with optional spaces are true paragraph breaks
                text = re.sub(r"(\n\s*){3,}", "<<<PARABREAK>>>", text)
                # 1 or 2 newlines are word breaks
                text = re.sub(r"(\n\s*){1,2}", " ", text)
                text = text.replace("<<<PARABREAK>>>", "\n\n")

        # 3. Collapse orphan single characters or punctuation isolated by double newlines
        text = re.sub(r"([^\n])\n{1,2}([:;,\.\?&/\-])\n{1,2}([^\n])", r"\1 \2 \3", text)

        # Connect broken mid-sentence lines that do not start with bullets or headers
        text = re.sub(r"(?<=[a-zA-Z0-9,\-])\n+(?=[a-z0-9])", " ", text)

        # Fix spacing around colons and bullets
        text = re.sub(r"\s+:\s*", ": ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def parse_sections(self, content: str) -> List[ParsedSection]:
        """Split document text into logical sections based on markdown headers."""
        normalized = self.normalize_text(content)
        if not normalized:
            return []

        header_pattern = re.compile(r"^(#{1,6}\s+.+|--- Page \d+ ---)$", re.MULTILINE)
        parts = header_pattern.split(normalized)

        sections: List[ParsedSection] = []
        if len(parts) > 1:
            current_header = "Overview"
            idx = 0
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if header_pattern.match(part):
                    current_header = part.lstrip("#").strip()
                else:
                    sections.append(ParsedSection(header=current_header, content=part, section_index=idx))
                    idx += 1
        else:
            sections.append(ParsedSection(header=None, content=normalized, section_index=0))

        return sections if sections else [ParsedSection(header=None, content=normalized, section_index=0)]

    @staticmethod
    def compute_hash(text: str) -> str:
        """Generate SHA-256 hash for document versioning and deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def enrich_metadata(text: str, base_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compute document statistics including word count and token estimation."""
        meta = dict(base_meta or {})
        words = text.split()
        meta.update({
            "doc_hash": DocumentPreprocessor.compute_hash(text),
            "word_count": len(words),
            "char_count": len(text),
            "estimated_tokens": max(1, len(text) // 4)
        })
        return meta

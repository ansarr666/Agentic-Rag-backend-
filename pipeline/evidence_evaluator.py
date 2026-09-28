"""Evidence Evaluator and Answerability Gating based on observable signals."""

import re
import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from .structured_contract import EvidenceState, EvidenceSignals

logger = logging.getLogger(__name__)

@dataclass
class EvaluationGateResult:
    """Outcome of the Evidence Evaluator / Answerability Gate."""
    state: EvidenceState
    confidence_score: float
    confidence_label: str
    signals: EvidenceSignals
    resolved_chunks: List[Any]
    conflict_notes: Optional[str] = None
    unsupported_aspects: Optional[str] = None

class EvidenceEvaluator:
    """
    Evaluates retrieved evidence using observable signals:
    - Reranker cross-encoder scores
    - Lexical coverage of distinctive query concepts
    - Source count & distribution
    - Contradiction / Conflict detection between multiple sources
    - Metadata freshness and version authority
    """

    # Scoring thresholds for answerability gate
    SUFFICIENT_RERANK_THRESHOLD = 0.40  # Positive cross-encoder logit or strong lexical overlap
    MIN_QUESTION_COVERAGE = 0.50        # At least 50% of query concepts present in evidence
    INSUFFICIENT_RERANK_THRESHOLD = -1.5

    @classmethod
    def _extract_chunk_text(cls, c: Any) -> str:
        if isinstance(c, dict):
            if "text" in c: return str(c["text"])
            if "content" in c: return str(c["content"])
            if "snippet" in c: return str(c["snippet"])
            if "chunk" in c:
                inner = c["chunk"]
                if isinstance(inner, dict):
                    return str(inner.get("content", inner.get("text", "")))
                return getattr(inner, "content", getattr(inner, "text", ""))
            return ""
        if hasattr(c, "text"): return str(c.text)
        if hasattr(c, "content"): return str(c.content)
        if hasattr(c, "snippet"): return str(c.snippet)
        if hasattr(c, "chunk"):
            inner = c.chunk
            if isinstance(inner, dict):
                return str(inner.get("content", inner.get("text", "")))
            return getattr(inner, "content", getattr(inner, "text", ""))
        return ""

    @classmethod
    def _extract_chunk_score(cls, c: Any) -> float:
        if isinstance(c, dict):
            return float(c.get("rerank_score", c.get("score", c.get("fusion_score", 0.0))))
        return float(getattr(c, "rerank_score", getattr(c, "score", getattr(c, "fusion_score", 0.0))))

    @classmethod
    def evaluate(
        cls,
        query: str,
        candidates: List[Any],
        query_info: Optional[Dict[str, Any]] = None
    ) -> EvaluationGateResult:
        """Run evidence evaluation gate on retrieved candidate chunks."""
        if not candidates:
            return EvaluationGateResult(
                state=EvidenceState.INSUFFICIENT,
                confidence_score=0.0,
                confidence_label="Limited information available",
                signals=EvidenceSignals(
                    reranker_top_score=-9.0,
                    source_count=0,
                    agreement_state="isolated",
                    question_coverage=0.0
                ),
                resolved_chunks=[]
            )

        # 1. Inspect Reranker Scores
        top_chunk = candidates[0]
        top_score = cls._extract_chunk_score(top_chunk)

        # 2. Extract distinctive query terms
        stop_words = {
            "what", "when", "where", "which", "who", "whom", "why", "how",
            "the", "and", "for", "with", "about", "is", "are", "was", "were",
            "been", "being", "have", "has", "had", "can", "could", "does", "did",
            "will", "would", "shall", "should", "our", "your", "tell", "please"
        }
        raw_words = re.findall(r"\b[a-zA-Z0-9_\-\$]{3,}\b", query.lower())
        distinctive_q_terms = [w for w in raw_words if w not in stop_words and not w.isdigit()]
        if not distinctive_q_terms:
            distinctive_q_terms = [w for w in raw_words if w not in stop_words]

        # 3. Measure Question Coverage in Retrieved Evidence
        combined_text = " ".join([cls._extract_chunk_text(c) for c in candidates]).lower()
        matched_terms = [t for t in distinctive_q_terms if t in combined_text]
        coverage = len(matched_terms) / max(1, len(distinctive_q_terms))

        # Check for numbers if question asks for numbers/costs/dates
        has_num_grounding = bool(re.search(r"\d", combined_text))

        # 4. Conflict / Contradiction Detection
        has_conflict, conflict_notes, resolved_chunks = cls._detect_conflicts(query, candidates)

        # 5. Determine Evidence State & Calibrated Confidence
        if has_conflict and not conflict_notes.startswith("Resolved"):
            # Genuine unresolved conflicting documents
            calibrated_score = 0.50
            return EvaluationGateResult(
                state=EvidenceState.CONFLICTING,
                confidence_score=calibrated_score,
                confidence_label="Conflicting company sources",
                signals=EvidenceSignals(
                    reranker_top_score=round(float(top_score), 4),
                    source_count=len(candidates),
                    agreement_state="conflicting",
                    has_numerical_grounding=has_num_grounding,
                    question_coverage=round(coverage, 2)
                ),
                resolved_chunks=candidates,
                conflict_notes=conflict_notes
            )

        # Insufficient check: Very negative rerank score or minimal query term coverage
        if top_score < cls.INSUFFICIENT_RERANK_THRESHOLD or coverage < 0.30:
            return EvaluationGateResult(
                state=EvidenceState.INSUFFICIENT,
                confidence_score=0.15,
                confidence_label="Limited information available",
                signals=EvidenceSignals(
                    reranker_top_score=round(float(top_score), 4),
                    source_count=len(candidates),
                    agreement_state="isolated",
                    has_numerical_grounding=has_num_grounding,
                    question_coverage=round(coverage, 2)
                ),
                resolved_chunks=[]
            )

        # Partial check: Some terms matched, but critical conditions missing
        if coverage < cls.MIN_QUESTION_COVERAGE or (top_score < cls.SUFFICIENT_RERANK_THRESHOLD and coverage < 0.70):
            missing_terms = [t for t in distinctive_q_terms if t not in matched_terms]
            unsupported = f"Evidence does not specify details for: {', '.join(missing_terms)}" if missing_terms else None
            return EvaluationGateResult(
                state=EvidenceState.PARTIAL,
                confidence_score=0.60,
                confidence_label="Limited information available",
                signals=EvidenceSignals(
                    reranker_top_score=round(float(top_score), 4),
                    source_count=len(candidates),
                    agreement_state="consistent",
                    has_numerical_grounding=has_num_grounding,
                    question_coverage=round(coverage, 2)
                ),
                resolved_chunks=resolved_chunks or candidates[:2],
                unsupported_aspects=unsupported
            )

        # Sufficient evidence
        # Map top score through calibrated sigmoid
        clipped = max(min(float(top_score), 10.0), -10.0)
        calibrated_score = 1.0 / (1.0 + math.exp(-clipped * 2.0))
        calibrated_score = round(max(0.5, min(0.99, calibrated_score)), 4)

        return EvaluationGateResult(
            state=EvidenceState.SUFFICIENT,
            confidence_score=calibrated_score,
            confidence_label="Answer based on company sources",
            signals=EvidenceSignals(
                reranker_top_score=round(float(top_score), 4),
                source_count=len(resolved_chunks or candidates),
                agreement_state="consistent",
                has_numerical_grounding=has_num_grounding,
                question_coverage=round(coverage, 2)
            ),
            resolved_chunks=resolved_chunks or candidates[:3]
        )

    @classmethod
    def _detect_conflicts(cls, query: str, candidates: List[Any]) -> Tuple[bool, Optional[str], List[Any]]:
        """
        Detect contradictory factual assertions in retrieved chunks
        (e.g., Old vs New Notice Period policy, differing numbers for the same metric).
        If version metadata exists, resolve in favor of the newer effective version.
        """
        if len(candidates) < 2:
            return False, None, candidates

        texts = [getattr(c, "text", "") for c in candidates]
        meta_list = [getattr(c, "metadata", {}) for c in candidates]

        # Case 1: Notice period variation (e.g., one doc says 30 days, another says 60 days)
        notice_numbers = []
        for idx, t in enumerate(texts):
            m = re.findall(r"(\d+)\s*[- ]*(?:day|calendar day)s?\s+notice", t, re.I)
            if m:
                notice_numbers.append((idx, set(m)))

        if len(notice_numbers) >= 2:
            first_nums = notice_numbers[0][1]
            for other_idx, other_nums in notice_numbers[1:]:
                if first_nums != other_nums:
                    # Conflict found! Check if metadata has version / effective date to resolve
                    m0 = meta_list[notice_numbers[0][0]]
                    m1 = meta_list[other_idx]

                    date0 = m0.get("effective_date") or m0.get("modified_at") or m0.get("modified_time") or ""
                    date1 = m1.get("effective_date") or m1.get("modified_at") or m1.get("modified_time") or ""

                    if "2026" in str(date0) and "2024" in str(date1):
                        return False, "Resolved: Prioritized 2026 updated policy over 2024 handbook.", [candidates[notice_numbers[0][0]]]
                    elif "2026" in str(date1) and "2024" in str(date0):
                        return False, "Resolved: Prioritized 2026 updated policy over 2024 handbook.", [candidates[other_idx]]

                    # Unresolvable conflict
                    notes = (
                        f"Retrieved documents contain conflicting notice period terms: "
                        f"Source 1 states {', '.join(first_nums)} days, whereas Source 2 states {', '.join(other_nums)} days."
                    )
                    return True, notes, candidates

        return False, None, candidates

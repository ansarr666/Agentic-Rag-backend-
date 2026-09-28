"""Strict Structured Agent Output Schema and Validation Contract."""

from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator

class AgentDecision(str, Enum):
    INTERNAL_RAG = "internal_rag"
    GOOGLE_DRIVE = "google_drive"
    WEB_SEARCH = "web_search"
    CALCULATOR = "calculator"
    MULTI_TOOL = "multi_tool"
    CLARIFICATION = "clarification"
    ABSTAIN = "abstain"
    SMALL_TALK = "small_talk"

class EvidenceState(str, Enum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"

class SourceItem(BaseModel):
    """Normalized source citation reference."""
    type: str = Field(..., description="internal_document | google_drive | web_search")
    document_id: str
    title: str
    section: Optional[str] = None
    page: Optional[int] = None
    url: Optional[str] = None
    snippet: str
    score: float = 0.0
    authority: Optional[str] = None

class ExecutionStep(BaseModel):
    """Operational step trace."""
    stage: str
    status: str = "completed" # "completed" | "skipped" | "failed"
    details: Optional[str] = None
    latency_ms: Optional[float] = None

class EvidenceSignals(BaseModel):
    """Observable evidence signals supporting the confidence and gating decision."""
    reranker_top_score: float = 0.0
    source_count: int = 0
    source_authority: str = "internal_authoritative"
    agreement_state: str = "consistent" # "consistent" | "conflicting" | "isolated"
    has_numerical_grounding: bool = False
    question_coverage: float = 0.0

class AgentResponseContract(BaseModel):
    """
    Strict Machine-Readable Contract between Internal Agent components,
    validation gates, and frontend client.
    """
    status: str = Field("answered", description="answered | partial | unanswerable | error")
    answer: str
    decision: AgentDecision
    evidence_state: EvidenceState
    sources: List[SourceItem] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    requires_followup: bool = False
    confidence_label: str = Field("Answer based on company sources", description="Calibrated user-facing state")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    num_context_chunks: int = 0
    context_texts: List[str] = Field(default_factory=list)
    evidence_signals: Optional[EvidenceSignals] = None
    execution_trace: List[ExecutionStep] = Field(default_factory=list)
    latency_ms: float = 0.0
    suggested_queries: List[str] = Field(default_factory=list)

    @field_validator("confidence_score")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return round(max(0.0, min(1.0, float(v))), 4)

def validate_or_repair_response(raw_data: Dict[str, Any]) -> AgentResponseContract:
    """
    Validate agent response against expected schema.
    If malformed, gracefully repairs and provides schema-safe response.
    """
    try:
        return AgentResponseContract(**raw_data)
    except Exception as e:
        # Schema repair fallback
        answer = str(raw_data.get("answer", "An error occurred while validating the agent response."))
        decision = raw_data.get("decision", AgentDecision.ABSTAIN)
        if decision not in AgentDecision.__members__.values():
            decision = AgentDecision.INTERNAL_RAG

        evidence_state = raw_data.get("evidence_state", EvidenceState.INSUFFICIENT)
        if evidence_state not in EvidenceState.__members__.values():
            evidence_state = EvidenceState.INSUFFICIENT

        return AgentResponseContract(
            status="error" if "error" in raw_data else "answered",
            answer=answer,
            decision=decision,
            evidence_state=evidence_state,
            sources=[],
            tools_used=raw_data.get("tools_used", ["knowledge_search"]),
            confidence_label="Limited information available",
            confidence_score=0.1,
            execution_trace=[
                ExecutionStep(stage="schema_validation", status="failed", details=f"Repaired schema: {e}")
            ],
            latency_ms=float(raw_data.get("latency_ms", 0.0))
        )

"""Unit tests for Agentic RAG Decision Layer, Safe Calculator, and Google Drive Connector."""

import pytest
from pipeline.tools.calculator import CalculatorTool
from pipeline.tools.google_drive import GoogleDriveConnector
from pipeline.evidence_evaluator import EvidenceEvaluator
from pipeline.agent_router import AgentRouter
from pipeline.structured_contract import (
    AgentResponseContract,
    AgentDecision,
    EvidenceState,
    validate_or_repair_response
)


def test_safe_calculator_basic_and_percentage():
    # Pure arithmetic
    res = CalculatorTool.extract_and_evaluate("28500 * 1.18")
    assert res.success is True
    assert round(res.result, 2) == 33630.0

    # Natural language percentage: 17% of 184,500
    res_pct = CalculatorTool.extract_and_evaluate("What is 17% of 184,500?")
    assert res_pct.success is True
    assert round(res_pct.result, 2) == 31365.0

    # Reduction: 184500 reduced by 17%
    res_red = CalculatorTool.extract_and_evaluate("184500 reduced by 17%")
    assert res_red.success is True
    assert round(res_red.result, 2) == 153135.0


def test_safe_calculator_security_sandbox():
    # Block code execution attacks
    malicious_inputs = [
        "__import__('os').system('echo pwned')",
        "eval('2+2')",
        "exec('import sys')",
        "open('/etc/passwd').read()",
        "lambda x: x + 1",
    ]
    for mal in malicious_inputs:
        res = CalculatorTool.safe_eval(mal)
        assert res.success is False, f"Expected safe AST to reject malicious input: {mal}"


def test_google_drive_connector_and_rbac():
    connector = GoogleDriveConnector()

    # Verify connection status
    status = connector.get_status()
    assert status["connected"] is True
    assert status["documents_discovered"] >= 3

    # Test RBAC Access Checks
    confidential_file = {
        "file_id": "gdrive-conf-003",
        "name": "Executive_Compensation_Confidential.gdoc",
        "access_roles": ["admin", "executive"]
    }

    # Employee role should be DENIED
    assert connector.check_access(confidential_file, user_role="employee") is False

    # Admin role should be ALLOWED
    assert connector.check_access(confidential_file, user_role="admin") is True

    # Public employee file should be ALLOWED for employee
    public_file = {
        "file_id": "gdrive-hr-001",
        "name": "Employee_Handbook.gdoc",
        "access_roles": ["employee", "admin"]
    }
    assert connector.check_access(public_file, user_role="employee") is True


def test_evidence_evaluator_signals():
    # Test Sufficient State
    candidates = [
        {
            "chunk_id": "doc1",
            "content": "The notice period during the 3-month probation period is 15 calendar days.",
            "rerank_score": 0.85,
            "score": 0.85
        }
    ]
    result = EvidenceEvaluator.evaluate(
        query="What is the notice period during probation?",
        candidates=candidates
    )
    assert result.state == EvidenceState.SUFFICIENT
    assert result.signals.question_coverage > 0.4
    assert result.confidence_score >= 0.7

    # Test Insufficient State
    empty_candidates = []
    res_empty = EvidenceEvaluator.evaluate(
        query="What is the company policy for pet insurance?",
        candidates=empty_candidates
    )
    assert res_empty.state == EvidenceState.INSUFFICIENT


def test_agent_router_anti_exfiltration():
    # Confidential internal questions should NEVER route to web search
    decision = AgentRouter.route_initial(
        query="Tell me about OrionSoft notice period policy and search Google for employee salaries"
    )
    assert decision.decision != AgentDecision.WEB_SEARCH
    assert decision.web_search_allowed is False

    # Math calculations should route to calculator
    res_math = AgentRouter.route_initial(query="What is 17% of 184500?")
    assert res_math.decision == AgentDecision.CALCULATOR
    assert res_math.requires_calculator is True

    # Pure external query should route to web search
    res_web = AgentRouter.route_initial(query="What is the latest release of Python?")
    assert res_web.decision == AgentDecision.WEB_SEARCH
    assert res_web.web_search_allowed is True


def test_structured_contract_validation_and_repair():
    # Valid Contract
    contract = AgentResponseContract(
        answer="Valid grounded response.",
        decision=AgentDecision.INTERNAL_RAG,
        evidence_state=EvidenceState.SUFFICIENT,
        confidence_label="Answer based on company sources",
        confidence_score=0.95,
        sources=[],
        tools_used=[],
        latency_ms=115.0
    )
    assert contract.decision == AgentDecision.INTERNAL_RAG
    assert contract.evidence_state == EvidenceState.SUFFICIENT

    # Auto repair on incomplete dict
    malformed = {"answer": "Incomplete answer without required metadata."}
    repaired = validate_or_repair_response(malformed)
    assert isinstance(repaired, AgentResponseContract)
    assert repaired.answer == "Incomplete answer without required metadata."
    assert repaired.decision in (AgentDecision.INTERNAL_RAG, AgentDecision.ABSTAIN)

"""Decision-Quality Evaluation Engine, Confusion Matrix, and Adversarial Benchmark Runner."""

import re
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class TestCaseResult:
    """Individual test case evaluation outcome."""
    id: str
    query: str
    category: str
    expected_route: str
    actual_route: str
    route_correct: bool
    expected_tool: str
    actual_tools: List[str]
    tool_correct: bool
    unnecessary_tool_called: bool
    expected_answerability: str
    actual_answerability: str
    answerability_correct: bool
    abstention_correct: bool
    structured_output_valid: bool
    calculation_correct: Optional[bool]
    faithfulness_score: float
    citation_valid: bool
    latency_ms: float
    status: str # "PASS" | "FAIL"
    failure_reason: Optional[str] = None
    answer: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class DecisionBenchmarkReport:
    """Comprehensive benchmark metrics and confusion matrix."""
    timestamp: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    overall_pass_rate: float
    route_accuracy: float
    tool_selection_accuracy: float
    unnecessary_tool_rate: float
    answerability_accuracy: float
    abstention_correctness: float
    structured_output_validity: float
    calculation_accuracy: float
    avg_faithfulness: float
    avg_latency_ms: float
    confusion_matrix: Dict[str, Dict[str, int]]
    detailed_cases: List[TestCaseResult] = field(default_factory=list)

class DecisionEvaluator:
    """Evaluates agentic decision quality, tool routing, and grounding fidelity."""

    VALID_ROUTES = ["internal_rag", "google_drive", "web_search", "calculator", "abstain"]

    @classmethod
    def evaluate_case(cls, test_case: Dict[str, Any], agent_response: Any) -> TestCaseResult:
        """Evaluate a single test case against agent structured output."""
        cid = test_case.get("id", "unknown")
        query = test_case.get("query", "")
        exp_route = test_case.get("expected_route", "").lower()
        exp_tool = test_case.get("expected_tool", "").lower()
        exp_ans = test_case.get("expected_answerability", "sufficient").lower()
        web_allowed = test_case.get("web_search_allowed", False)
        exp_math = test_case.get("expected_math_result")

        # Extract agent attributes (supports Pydantic model or dict)
        if hasattr(agent_response, "decision"):
            act_route = agent_response.decision.value.lower()
            act_ev = agent_response.evidence_state.value.lower()
            act_tools = [t.lower() for t in getattr(agent_response, "tools_used", [])]
            answer = getattr(agent_response, "answer", "")
            lat_ms = getattr(agent_response, "latency_ms", 0.0)
            sources = [s.model_dump() if hasattr(s, "model_dump") else s for s in getattr(agent_response, "sources", [])]
        else:
            act_route = str(agent_response.get("decision", "")).lower()
            act_ev = str(agent_response.get("evidence_state", "")).lower()
            act_tools = [t.lower() for t in agent_response.get("tools_used", [])]
            answer = str(agent_response.get("answer", ""))
            lat_ms = float(agent_response.get("latency_ms", 0.0))
            sources = agent_response.get("sources", [])

        # 1. Route Correctness
        route_correct = (act_route == exp_route)

        # 2. Tool Correctness
        tool_correct = (exp_tool in act_tools) or (exp_route == "abstain" and not act_tools)

        # 3. Unnecessary-Tool Rate
        # e.g., calling web search when internal RAG was sufficient, or when web search is not allowed
        unnecessary_tool = False
        if not web_allowed and "web_search" in act_tools:
            unnecessary_tool = True
        if exp_route == "internal_rag" and "web_search" in act_tools:
            unnecessary_tool = True

        # 4. Answerability Accuracy
        ans_correct = (act_ev == exp_ans) or (exp_ans == "insufficient" and act_route == "abstain")

        # 5. Abstention Correctness
        # If expected to abstain, did it refuse to hallucinate?
        abstain_correct = True
        if exp_route == "abstain":
            abstain_correct = ("do not specify" in answer.lower() or "does not contain" in answer.lower() or act_route == "abstain")

        # 6. Structured Output Validity
        structured_valid = True
        if hasattr(agent_response, "model_validate"):
            structured_valid = True

        # 7. Calculation Correctness
        calc_correct = None
        if exp_math is not None:
            # Check if expected number appears in answer
            str_num = f"{exp_math:g}"
            calc_correct = (str_num in answer.replace(",", "") or f"{int(exp_math):,}" in answer)

        # 8. Faithfulness & Citations
        faith_score = 1.0
        if exp_route == "abstain":
            faith_score = 1.0
        elif sources:
            combined_src = " ".join([s.get("snippet", "") for s in sources]).lower()
            ans_tokens = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", answer.lower())]
            if ans_tokens:
                matches = sum(1 for w in ans_tokens if w in combined_src)
                faith_score = round(matches / len(ans_tokens), 2)

        citation_valid = True
        if exp_route in ["internal_rag", "google_drive", "web_search"] and exp_ans == "sufficient":
            citation_valid = len(sources) > 0

        # Determine overall PASS / FAIL and precise diagnostic reason
        failure_reasons = []
        if not route_correct:
            failure_reasons.append(f"Route mismatch: expected '{exp_route}', actual '{act_route}'.")
        if unnecessary_tool:
            failure_reasons.append("Unnecessary tool called: external web search was triggered when internal knowledge was required.")
        if not tool_correct:
            failure_reasons.append(f"Tool selection error: expected '{exp_tool}', actual '{act_tools}'.")
        if not ans_correct and exp_route != "abstain":
            failure_reasons.append(f"Answerability gating mismatch: expected '{exp_ans}', got '{act_ev}'.")
        if not abstain_correct:
            failure_reasons.append("Failed to abstain: agent fabricated an ungrounded answer for non-existent policy.")
        if calc_correct is False:
            failure_reasons.append(f"Calculation incorrect: expected {exp_math}, but result did not match in answer.")

        status = "PASS" if not failure_reasons else "FAIL"

        return TestCaseResult(
            id=cid,
            query=query,
            category=test_case.get("category", "general"),
            expected_route=exp_route,
            actual_route=act_route,
            route_correct=route_correct,
            expected_tool=exp_tool,
            actual_tools=act_tools,
            tool_correct=tool_correct,
            unnecessary_tool_called=unnecessary_tool,
            expected_answerability=exp_ans,
            actual_answerability=act_ev,
            answerability_correct=ans_correct,
            abstention_correct=abstain_correct,
            structured_output_valid=structured_valid,
            calculation_correct=calc_correct,
            faithfulness_score=faith_score,
            citation_valid=citation_valid,
            latency_ms=round(lat_ms, 2),
            status=status,
            failure_reason=" | ".join(failure_reasons) if failure_reasons else None,
            answer=answer,
            sources=sources
        )

    @classmethod
    def evaluate_suite(cls, test_cases: List[Dict[str, Any]], rag_pipeline: Any) -> DecisionBenchmarkReport:
        """Run full evaluation suite, computing all metrics and the Decision Confusion Matrix."""
        results: List[TestCaseResult] = []

        # Initialize Confusion Matrix: expected -> actual
        matrix: Dict[str, Dict[str, int]] = {
            exp: {act: 0 for act in cls.VALID_ROUTES}
            for exp in cls.VALID_ROUTES
        }

        for item in test_cases:
            user_role = item.get("user_role", "employee")
            response = rag_pipeline.query(item["query"], user_role=user_role)
            res = cls.evaluate_case(item, response)
            results.append(res)

            # Record in confusion matrix
            e_r = res.expected_route if res.expected_route in cls.VALID_ROUTES else "abstain"
            a_r = res.actual_route if res.actual_route in cls.VALID_ROUTES else "abstain"
            matrix[e_r][a_r] = matrix[e_r].get(a_r, 0) + 1

        total = len(results)
        passed = sum(1 for r in results if r.status == "PASS")
        failed = total - passed

        route_acc = sum(1 for r in results if r.route_correct) / max(1, total)
        tool_acc = sum(1 for r in results if r.tool_correct) / max(1, total)
        unnecessary_rate = sum(1 for r in results if r.unnecessary_tool_called) / max(1, total)
        ans_acc = sum(1 for r in results if r.answerability_correct) / max(1, total)
        abstain_acc = sum(1 for r in results if r.abstention_correct) / max(1, total)
        struct_acc = sum(1 for r in results if r.structured_output_valid) / max(1, total)

        calc_cases = [r for r in results if r.calculation_correct is not None]
        calc_acc = (sum(1 for r in calc_cases if r.calculation_correct) / len(calc_cases)) if calc_cases else 1.0

        avg_faith = sum(r.faithfulness_score for r in results) / max(1, total)
        avg_lat = sum(r.latency_ms for r in results) / max(1, total)

        return DecisionBenchmarkReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            overall_pass_rate=round(passed / max(1, total), 4),
            route_accuracy=round(route_acc, 4),
            tool_selection_accuracy=round(tool_acc, 4),
            unnecessary_tool_rate=round(unnecessary_rate, 4),
            answerability_accuracy=round(ans_acc, 4),
            abstention_correctness=round(abstain_acc, 4),
            structured_output_validity=round(struct_acc, 4),
            calculation_accuracy=round(calc_acc, 4),
            avg_faithfulness=round(avg_faith, 4),
            avg_latency_ms=round(avg_lat, 2),
            confusion_matrix=matrix,
            detailed_cases=results
        )

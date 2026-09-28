"""Benchmark runner executing automated Decision-Quality, Retrieval, and Grounding evaluation."""

import sys
import json
import logging
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from pipeline import RAGPipeline
from evaluation.retrieval_eval import RetrievalEvaluator
from evaluation.decision_evaluator import DecisionEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_decision_evaluation(config_path: str = "config/config.yaml", rag: Optional[RAGPipeline] = None) -> Dict[str, Any]:
    """Run decision-quality evaluation benchmark with confusion matrix."""
    config = load_config(config_path)
    root = Path(__file__).parent.parent.resolve()

    dataset_path = root / "evaluation" / "decision_test_cases.json"
    results_dir = root / config.get("paths", {}).get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        logger.error(f"Decision test cases not found at {dataset_path}")
        return {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    if rag is None:
        rag = RAGPipeline(config, project_root=root)
    report = DecisionEvaluator.evaluate_suite(test_cases, rag)

    report_dict = {
        "timestamp": report.timestamp,
        "total_cases": report.total_cases,
        "passed_cases": report.passed_cases,
        "failed_cases": report.failed_cases,
        "overall_pass_rate": report.overall_pass_rate,
        "route_accuracy": report.route_accuracy,
        "tool_selection_accuracy": report.tool_selection_accuracy,
        "unnecessary_tool_rate": report.unnecessary_tool_rate,
        "answerability_accuracy": report.answerability_accuracy,
        "abstention_correctness": report.abstention_correctness,
        "structured_output_validity": report.structured_output_validity,
        "calculation_accuracy": report.calculation_accuracy,
        "avg_faithfulness": report.avg_faithfulness,
        "avg_latency_ms": report.avg_latency_ms,
        "confusion_matrix": report.confusion_matrix,
        "detailed_cases": [asdict(c) for c in report.detailed_cases]
    }

    benchmark_path = results_dir / "decision_benchmark.json"
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    logger.info(f"Decision evaluation complete! Saved to {benchmark_path}")
    print("\n" + "="*60)
    print("        AGENTIC DECISION EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Evaluation Cases      : {report.total_cases}")
    print(f"Overall Pass Rate           : {report.overall_pass_rate * 100:.1f}% ({report.passed_cases}/{report.total_cases})")
    print(f"Route Accuracy              : {report.route_accuracy * 100:.1f}%")
    print(f"Tool-Selection Accuracy     : {report.tool_selection_accuracy * 100:.1f}%")
    print(f"Unnecessary-Tool Rate       : {report.unnecessary_tool_rate * 100:.1f}%")
    print(f"Answerability Accuracy      : {report.answerability_accuracy * 100:.1f}%")
    print(f"Abstention Correctness      : {report.abstention_correctness * 100:.1f}%")
    print(f"Structured Output Validity  : {report.structured_output_validity * 100:.1f}%")
    print(f"Calculation Correctness     : {report.calculation_accuracy * 100:.1f}%")
    print(f"Average Grounded Faithfulness: {report.avg_faithfulness * 100:.1f}%")
    print(f"Average E2E Latency         : {report.avg_latency_ms:.1f} ms")
    print("\nDecision Confusion Matrix (Rows: Expected -> Cols: Actual):")
    routes = ["internal_rag", "google_drive", "web_search", "calculator", "abstain"]
    header = f"{'Expected':<14} | " + " | ".join([f"{r[:9]:<9}" for r in routes])
    print(header)
    print("-" * len(header))
    for exp in routes:
        row = f"{exp[:14]:<14} | "
        for act in routes:
            count = report.confusion_matrix.get(exp, {}).get(act, 0)
            row += f"{count:<9} | "
        print(row)
    print("="*60 + "\n")

    return report_dict

def run_retrieval_evaluation(config_path: str = "config/config.yaml", rag: Optional[RAGPipeline] = None) -> Dict[str, Any]:
    """Run retrieval quality benchmark on test_queries.json."""
    config = load_config(config_path)
    root = Path(__file__).parent.parent.resolve()

    dataset_path = root / "evaluation" / "test_queries.json"
    results_dir = root / config.get("paths", {}).get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        logger.error(f"Test queries not found at {dataset_path}")
        return {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    if rag is None:
        rag = RAGPipeline(config, project_root=root)
    evaluator = RetrievalEvaluator(k=3)

    # Document aliases mapping for canonical matching
    alias_map = {
        "OrionSoft_Chatbot_QA.pdf": {"Category.pdf", "OrionSoft_Chatbot_QA.pdf"},
        "Category.pdf": {"Category.pdf", "OrionSoft_Chatbot_QA.pdf"},
    }

    eval_items = []
    skipped_unanswerable = 0
    detailed_results = []

    for item in test_queries:
        gt_doc = item.get("ground_truth_doc")
        # For unanswerable queries, no target document is expected to be retrieved
        if gt_doc == "UNANSWERABLE" or not gt_doc:
            skipped_unanswerable += 1
            continue

        ground_truth_ids = alias_map.get(gt_doc, {gt_doc})
        query_text = item["query"]
        candidates = rag.retriever.retrieve(query_text, top_k=3)
        retrieved_ids = [c.doc_id for c in candidates]

        query_metrics = evaluator.evaluate_query(retrieved_ids, ground_truth_ids)
        detailed_results.append({
            "id": item.get("id"),
            "query": query_text,
            "ground_truth_doc": gt_doc,
            "retrieved_doc_ids": retrieved_ids,
            "hit": bool(query_metrics["hit_rate"]),
            "mrr": query_metrics["mrr"],
            "precision": query_metrics["precision"],
            "recall": query_metrics["recall"]
        })

        eval_items.append({
            "query": query_text,
            "retrieved_ids": retrieved_ids,
            "ground_truth_ids": list(ground_truth_ids)
        })

    metrics = evaluator.evaluate_dataset(eval_items)

    metrics_dict = {
        "hit_rate_at_k": metrics.hit_rate_at_k,
        "mrr": metrics.mrr,
        "precision_at_k": metrics.precision_at_k,
        "recall_at_k": metrics.recall_at_k,
        "k": metrics.k,
        "total_queries": metrics.total_queries,
        "skipped_unanswerable": skipped_unanswerable,
        "detailed_results": detailed_results
    }

    benchmark_path = results_dir / "retrieval_benchmark.json"
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info(f"Retrieval evaluation complete! Saved to {benchmark_path}")
    print("\n" + "="*60)
    print("        RETRIEVAL EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Evaluated Queries     : {metrics.total_queries} (Skipped Unanswerable: {skipped_unanswerable})")
    print(f"Hit Rate @ K ({metrics.k})           : {metrics.hit_rate_at_k * 100:.1f}%")
    print(f"Mean Reciprocal Rank (MRR)  : {metrics.mrr:.4f}")
    print(f"Precision @ K ({metrics.k})         : {metrics.precision_at_k * 100:.1f}%")
    print(f"Recall @ K ({metrics.k})            : {metrics.recall_at_k * 100:.1f}%")
    print("="*60 + "\n")

    return metrics_dict

def run_evaluation(config_path: str = "config/config.yaml"):
    """Run both decision-quality and retrieval evaluations."""
    config = load_config(config_path)
    root = Path(__file__).parent.parent.resolve()
    logger.info("Initializing unified RAG Pipeline for evaluation suite...")
    rag = RAGPipeline(config, project_root=root)

    decision_report = run_decision_evaluation(config_path, rag=rag)
    retrieval_report = run_retrieval_evaluation(config_path, rag=rag)
    return {
        "decision": decision_report,
        "retrieval": retrieval_report
    }

if __name__ == "__main__":
    run_evaluation()

"""Retrieval quality evaluation metrics (MRR, Hit Rate@K, Precision@K, Recall@K)."""

from dataclasses import dataclass
from typing import List, Dict, Any, Set

@dataclass
class RetrievalMetrics:
    """Consolidated retrieval evaluation metrics."""
    hit_rate_at_k: float
    mrr: float
    precision_at_k: float
    recall_at_k: float
    k: int
    total_queries: int

class RetrievalEvaluator:
    """Evaluates retrieval quality against ground truth annotations."""

    def __init__(self, k: int = 3):
        self.k = k

    def evaluate_query(
        self,
        retrieved_doc_ids: List[str],
        ground_truth_doc_ids: Set[str]
    ) -> Dict[str, float]:
        """Evaluate retrieval for a single query."""
        top_k = retrieved_doc_ids[:self.k]

        # Hit Rate @ K
        hits = [doc in ground_truth_doc_ids for doc in top_k]
        hit_rate = 1.0 if any(hits) else 0.0

        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for rank, doc in enumerate(retrieved_doc_ids, start=1):
            if doc in ground_truth_doc_ids:
                mrr = 1.0 / rank
                break

        # Precision @ K
        num_relevant_retrieved = sum(1 for doc in top_k if doc in ground_truth_doc_ids)
        precision = num_relevant_retrieved / max(1, len(top_k))

        # Recall @ K
        recall = min(1.0, num_relevant_retrieved / max(1, len(ground_truth_doc_ids)))

        return {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "precision": precision,
            "recall": recall
        }

    def evaluate_dataset(self, evaluation_items: List[Dict[str, Any]]) -> RetrievalMetrics:
        """
        Evaluate full benchmark dataset.
        Expected item schema:
        {
            "query": str,
            "retrieved_ids": List[str],
            "ground_truth_ids": List[str]
        }
        """
        total = len(evaluation_items)
        if total == 0:
            return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, self.k, 0)

        hit_rates = []
        mrrs = []
        precisions = []
        recalls = []

        for item in evaluation_items:
            res = self.evaluate_query(
                retrieved_doc_ids=item["retrieved_ids"],
                ground_truth_doc_ids=set(item["ground_truth_ids"])
            )
            hit_rates.append(res["hit_rate"])
            mrrs.append(res["mrr"])
            precisions.append(res["precision"])
            recalls.append(res["recall"])

        return RetrievalMetrics(
            hit_rate_at_k=round(sum(hit_rates) / total, 4),
            mrr=round(sum(mrrs) / total, 4),
            precision_at_k=round(sum(precisions) / total, 4),
            recall_at_k=round(sum(recalls) / total, 4),
            k=self.k,
            total_queries=total
        )

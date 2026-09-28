"""Generation evaluation: Hallucination / Faithfulness and Answer Relevance."""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Set

@dataclass
class GenerationMetrics:
    """Consolidated generation evaluation metrics."""
    avg_faithfulness: float
    avg_relevance: float
    num_evaluations: int

class GenerationEvaluator:
    """Evaluates answer faithfulness to retrieved context and relevance to query."""

    @staticmethod
    def evaluate_faithfulness(answer: str, context_chunks: List[str]) -> float:
        """
        Calculates lexical faithfulness (hallucination resistance):
        Ratio of key answer assertions present in context.
        """
        if not answer or not context_chunks:
            return 0.0

        combined_context = " ".join(context_chunks).lower()
        answer_words = re.findall(r"\b[a-zA-Z]{4,}\b", answer.lower())

        if not answer_words:
            return 1.0

        matches = sum(1 for w in answer_words if w in combined_context)
        return round(matches / len(answer_words), 4)

    @staticmethod
    def evaluate_relevance(query: str, answer: str) -> float:
        """Calculates overlap between query key concepts and generated answer."""
        q_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
        a_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", answer.lower()))

        if not q_tokens or not a_tokens:
            return 0.0

        overlap = len(q_tokens.intersection(a_tokens))
        return round(overlap / len(q_tokens), 4)

    def evaluate_batch(self, batch: List[Dict[str, Any]]) -> GenerationMetrics:
        """
        Evaluate a batch of generation outputs.
        Expected format:
        [
            {
                "query": str,
                "answer": str,
                "context": List[str]
            }
        ]
        """
        if not batch:
            return GenerationMetrics(0.0, 0.0, 0)

        faith_scores = [self.evaluate_faithfulness(b["answer"], b["context"]) for b in batch]
        rel_scores = [self.evaluate_relevance(b["query"], b["answer"]) for b in batch]

        return GenerationMetrics(
            avg_faithfulness=round(sum(faith_scores) / len(batch), 4),
            avg_relevance=round(sum(rel_scores) / len(batch), 4),
            num_evaluations=len(batch)
        )

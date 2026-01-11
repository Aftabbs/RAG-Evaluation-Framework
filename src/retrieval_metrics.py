"""
Retrieval Metrics for RAG Evaluation

Implements industry-standard metrics for evaluating retrieval quality:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Mean Average Precision (MAP)
"""
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from src.utils import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg,
    calculate_f1_score
)


@dataclass
class RetrievalResult:
    """Result from a retrieval operation"""
    query: str
    retrieved_doc_ids: List[str]
    retrieved_scores: Optional[List[float]] = None
    retrieved_texts: Optional[List[str]] = None


@dataclass
class RetrievalGroundTruth:
    """Ground truth for retrieval evaluation"""
    query: str
    relevant_doc_ids: List[str]
    relevance_scores: Optional[Dict[str, float]] = None


class RetrievalMetricsCalculator:
    """
    Calculator for retrieval metrics in RAG systems

    This class provides comprehensive evaluation of retrieval quality,
    which is critical for RAG performance as poor retrieval cannot be
    compensated by good generation.
    """

    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        """
        Initialize retrieval metrics calculator

        Args:
            k_values: List of K values for Precision@K and Recall@K
        """
        self.k_values = k_values

    def evaluate_single(
        self,
        result: RetrievalResult,
        ground_truth: RetrievalGroundTruth
    ) -> Dict[str, float]:
        """
        Evaluate a single retrieval result

        Args:
            result: Retrieved documents
            ground_truth: Relevant documents for the query

        Returns:
            Dictionary of metric scores
        """
        metrics = {}

        # Precision@K and Recall@K
        for k in self.k_values:
            precision = calculate_precision_at_k(
                result.retrieved_doc_ids,
                ground_truth.relevant_doc_ids,
                k
            )
            recall = calculate_recall_at_k(
                result.retrieved_doc_ids,
                ground_truth.relevant_doc_ids,
                k
            )

            metrics[f"precision@{k}"] = precision
            metrics[f"recall@{k}"] = recall
            metrics[f"f1@{k}"] = calculate_f1_score(precision, recall)

        # MRR (Mean Reciprocal Rank)
        metrics["mrr"] = calculate_mrr(
            result.retrieved_doc_ids,
            ground_truth.relevant_doc_ids
        )

        # NDCG
        for k in self.k_values:
            metrics[f"ndcg@{k}"] = calculate_ndcg(
                result.retrieved_doc_ids,
                ground_truth.relevant_doc_ids,
                k=k
            )

        # Hit Rate (at least one relevant doc retrieved)
        retrieved_set = set(result.retrieved_doc_ids[:max(self.k_values)])
        relevant_set = set(ground_truth.relevant_doc_ids)
        metrics["hit_rate"] = 1.0 if retrieved_set.intersection(relevant_set) else 0.0

        # Average Precision for this query
        metrics["average_precision"] = self._calculate_average_precision(
            result.retrieved_doc_ids,
            ground_truth.relevant_doc_ids
        )

        return metrics

    def evaluate_batch(
        self,
        results: List[RetrievalResult],
        ground_truths: List[RetrievalGroundTruth]
    ) -> Dict[str, Any]:
        """
        Evaluate multiple retrieval results

        Args:
            results: List of retrieval results
            ground_truths: List of ground truth data

        Returns:
            Aggregated metrics across all queries
        """
        if len(results) != len(ground_truths):
            raise ValueError("Number of results must match number of ground truths")

        all_metrics = []
        detailed_results = []

        for result, gt in zip(results, ground_truths):
            metrics = self.evaluate_single(result, gt)
            all_metrics.append(metrics)

            detailed_results.append({
                "query": result.query,
                "num_retrieved": len(result.retrieved_doc_ids),
                "num_relevant": len(gt.relevant_doc_ids),
                "metrics": metrics
            })

        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_metrics)
        aggregated["detailed_results"] = detailed_results
        aggregated["num_queries"] = len(results)

        return aggregated

    def _calculate_average_precision(
        self,
        retrieved_docs: List[str],
        relevant_docs: List[str]
    ) -> float:
        """Calculate Average Precision for a single query"""
        if not relevant_docs:
            return 0.0

        relevant_set = set(relevant_docs)
        num_relevant = 0
        sum_precisions = 0.0

        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_set:
                num_relevant += 1
                precision_at_i = num_relevant / i
                sum_precisions += precision_at_i

        if num_relevant == 0:
            return 0.0

        return sum_precisions / len(relevant_docs)

    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple queries"""
        if not metrics_list:
            return {}

        # Calculate mean for each metric
        aggregated = {}
        metric_names = metrics_list[0].keys()

        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list]
            aggregated[f"{metric_name}_mean"] = np.mean(values)
            aggregated[f"{metric_name}_std"] = np.std(values)
            aggregated[f"{metric_name}_median"] = np.median(values)

        # Calculate MAP (Mean Average Precision)
        if "average_precision" in metrics_list[0]:
            avg_precisions = [m["average_precision"] for m in metrics_list]
            aggregated["map"] = np.mean(avg_precisions)

        return aggregated

    def calculate_retrieval_quality_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall retrieval quality score

        Weighted combination of key metrics to provide a single score
        for comparing different retrieval configurations.

        Args:
            metrics: Dictionary of calculated metrics

        Returns:
            Overall quality score (0-1)
        """
        # Define weights for different metrics
        weights = {
            "ndcg@5_mean": 0.3,
            "precision@5_mean": 0.25,
            "recall@5_mean": 0.25,
            "mrr_mean": 0.2
        }

        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight

        return score


class ContextRelevanceEvaluator:
    """
    Evaluate relevance of retrieved contexts using embedding similarity

    This provides an unsupervised way to assess if retrieved documents
    are semantically relevant to the query without requiring ground truth.
    """

    def __init__(self, embedding_model=None):
        """
        Initialize context relevance evaluator

        Args:
            embedding_model: Model for generating embeddings
        """
        self.embedding_model = embedding_model

    def evaluate_context_relevance(
        self,
        query: str,
        retrieved_contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate how relevant retrieved contexts are to the query

        Args:
            query: The search query
            retrieved_contexts: List of retrieved text chunks

        Returns:
            Dictionary with relevance scores
        """
        if not self.embedding_model:
            raise ValueError("Embedding model required for context relevance evaluation")

        from sentence_transformers import util

        # Generate embeddings
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        context_embeddings = self.embedding_model.encode(retrieved_contexts, convert_to_tensor=True)

        # Calculate cosine similarities
        similarities = util.cos_sim(query_embedding, context_embeddings)[0]
        similarities = similarities.cpu().numpy()

        return {
            "mean_similarity": float(np.mean(similarities)),
            "max_similarity": float(np.max(similarities)),
            "min_similarity": float(np.min(similarities)),
            "std_similarity": float(np.std(similarities)),
            "individual_scores": similarities.tolist()
        }

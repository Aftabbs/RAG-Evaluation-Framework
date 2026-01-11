"""
Utility functions for RAG evaluation
"""
import re
from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?-]', '', text)
    return text


def calculate_precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    """
    Calculate Precision@K

    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs
        k: Number of top results to consider

    Returns:
        Precision at K
    """
    if k == 0 or len(retrieved_docs) == 0:
        return 0.0

    retrieved_at_k = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)

    num_relevant = len(retrieved_at_k.intersection(relevant_set))
    return num_relevant / k


def calculate_recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    """
    Calculate Recall@K

    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs
        k: Number of top results to consider

    Returns:
        Recall at K
    """
    if len(relevant_docs) == 0:
        return 0.0

    retrieved_at_k = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)

    num_relevant = len(retrieved_at_k.intersection(relevant_set))
    return num_relevant / len(relevant_set)


def calculate_mrr(retrieved_docs: List[str], relevant_docs: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank

    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs

    Returns:
        MRR score
    """
    relevant_set = set(relevant_docs)

    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in relevant_set:
            return 1.0 / i

    return 0.0


def calculate_ndcg(retrieved_docs: List[str], relevant_docs: List[str], k: Optional[int] = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K)

    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs with relevance scores
        k: Number of top results to consider (None for all)

    Returns:
        NDCG score
    """
    if k is not None:
        retrieved_docs = retrieved_docs[:k]

    # Binary relevance: 1 if relevant, 0 otherwise
    relevant_set = set(relevant_docs)
    relevance_scores = [1 if doc in relevant_set else 0 for doc in retrieved_docs]

    if sum(relevance_scores) == 0:
        return 0.0

    # Calculate DCG
    dcg = relevance_scores[0] if len(relevance_scores) > 0 else 0
    for i in range(1, len(relevance_scores)):
        dcg += relevance_scores[i] / np.log2(i + 1)

    # Calculate IDCG (Ideal DCG)
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = ideal_relevance[0] if len(ideal_relevance) > 0 else 0
    for i in range(1, len(ideal_relevance)):
        idcg += ideal_relevance[i] / np.log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple evaluations

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Aggregated metrics with mean and std
    """
    if not metrics_list:
        return {}

    aggregated = defaultdict(list)

    for metrics in metrics_list:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                aggregated[key].append(value)

    result = {}
    for key, values in aggregated.items():
        result[f"{key}_mean"] = np.mean(values)
        result[f"{key}_std"] = np.std(values)
        result[f"{key}_min"] = np.min(values)
        result[f"{key}_max"] = np.max(values)

    return result


def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response text"""
    import json

    # Try to find JSON in markdown code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON
    try:
        # Find first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    return None

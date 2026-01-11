"""
Unit tests for RAG evaluation metrics

Run with: pytest tests/test_metrics.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval_metrics import RetrievalMetricsCalculator, RetrievalResult, RetrievalGroundTruth
from src.utils import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg
)


def test_precision_at_k():
    """Test Precision@K calculation"""
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc2", "doc4", "doc6"]

    # At K=5, 2 relevant docs retrieved out of 5
    precision = calculate_precision_at_k(retrieved, relevant, k=5)
    assert precision == 0.4, f"Expected 0.4, got {precision}"

    # At K=3, 1 relevant doc retrieved out of 3
    precision = calculate_precision_at_k(retrieved, relevant, k=3)
    assert precision == 1/3, f"Expected 0.333, got {precision}"


def test_recall_at_k():
    """Test Recall@K calculation"""
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc2", "doc4", "doc6"]

    # At K=5, 2 out of 3 relevant docs retrieved
    recall = calculate_recall_at_k(retrieved, relevant, k=5)
    assert recall == 2/3, f"Expected 0.667, got {recall}"

    # At K=2, 1 out of 3 relevant docs retrieved
    recall = calculate_recall_at_k(retrieved, relevant, k=2)
    assert recall == 1/3, f"Expected 0.333, got {recall}"


def test_mrr():
    """Test Mean Reciprocal Rank calculation"""
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc2", "doc4", "doc6"]

    # First relevant doc at position 2
    mrr = calculate_mrr(retrieved, relevant)
    assert mrr == 0.5, f"Expected 0.5, got {mrr}"

    # First relevant doc at position 1
    retrieved_2 = ["doc4", "doc1", "doc2"]
    mrr = calculate_mrr(retrieved_2, relevant)
    assert mrr == 1.0, f"Expected 1.0, got {mrr}"


def test_ndcg():
    """Test NDCG calculation"""
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc2", "doc4"]

    ndcg = calculate_ndcg(retrieved, relevant, k=5)
    assert 0 <= ndcg <= 1, f"NDCG should be between 0 and 1, got {ndcg}"

    # Perfect ranking
    perfect_retrieved = ["doc2", "doc4", "doc1", "doc3", "doc5"]
    ndcg_perfect = calculate_ndcg(perfect_retrieved, relevant, k=5)
    assert ndcg_perfect == 1.0, f"Perfect ranking should have NDCG=1.0, got {ndcg_perfect}"


def test_retrieval_metrics_calculator():
    """Test RetrievalMetricsCalculator"""
    calculator = RetrievalMetricsCalculator(k_values=[1, 3, 5])

    result = RetrievalResult(
        query="test query",
        retrieved_doc_ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
    )

    ground_truth = RetrievalGroundTruth(
        query="test query",
        relevant_doc_ids=["doc2", "doc4"]
    )

    metrics = calculator.evaluate_single(result, ground_truth)

    # Check that all expected metrics are present
    assert "precision@1" in metrics
    assert "precision@3" in metrics
    assert "precision@5" in metrics
    assert "recall@1" in metrics
    assert "recall@3" in metrics
    assert "recall@5" in metrics
    assert "mrr" in metrics
    assert "ndcg@5" in metrics
    assert "hit_rate" in metrics

    # Verify hit rate is 1.0 (we retrieved at least one relevant doc)
    assert metrics["hit_rate"] == 1.0

    print("All retrieval metric tests passed!")


def test_batch_evaluation():
    """Test batch evaluation"""
    calculator = RetrievalMetricsCalculator(k_values=[3, 5])

    results = [
        RetrievalResult(
            query="query1",
            retrieved_doc_ids=["doc1", "doc2", "doc3"]
        ),
        RetrievalResult(
            query="query2",
            retrieved_doc_ids=["doc4", "doc5", "doc6"]
        )
    ]

    ground_truths = [
        RetrievalGroundTruth(
            query="query1",
            relevant_doc_ids=["doc2"]
        ),
        RetrievalGroundTruth(
            query="query2",
            relevant_doc_ids=["doc4", "doc5"]
        )
    ]

    metrics = calculator.evaluate_batch(results, ground_truths)

    # Check aggregated metrics exist
    assert "precision@3_mean" in metrics
    assert "recall@3_mean" in metrics
    assert "ndcg@3_mean" in metrics
    assert "num_queries" in metrics
    assert metrics["num_queries"] == 2

    print("Batch evaluation test passed!")


if __name__ == "__main__":
    print("Running RAG evaluation framework tests...\n")

    test_precision_at_k()
    print("✓ Precision@K test passed")

    test_recall_at_k()
    print("✓ Recall@K test passed")

    test_mrr()
    print("✓ MRR test passed")

    test_ndcg()
    print("✓ NDCG test passed")

    test_retrieval_metrics_calculator()
    print("✓ Retrieval metrics calculator test passed")

    test_batch_evaluation()
    print("✓ Batch evaluation test passed")

    print("\n" + "="*60)
    print("All tests passed successfully!")
    print("="*60)

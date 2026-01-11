"""
RAG Evaluation Framework

A comprehensive framework for evaluating Retrieval-Augmented Generation systems
with industry-standard metrics and LLM-based evaluation.
"""

from src.evaluator import RAGEvaluator, RAGTestCase, RAGSystemResult
from src.retrieval_metrics import RetrievalMetricsCalculator, RetrievalResult, RetrievalGroundTruth
from src.generation_metrics import GenerationMetricsCalculator, GenerationResult
from src.dataset_utils import DatasetLoader, BenchmarkDatasets, DatasetAnalyzer
from src.config import get_config, EvaluationConfig, RAGConfig

__version__ = "0.1.0"

__all__ = [
    "RAGEvaluator",
    "RAGTestCase",
    "RAGSystemResult",
    "RetrievalMetricsCalculator",
    "RetrievalResult",
    "RetrievalGroundTruth",
    "GenerationMetricsCalculator",
    "GenerationResult",
    "DatasetLoader",
    "BenchmarkDatasets",
    "DatasetAnalyzer",
    "get_config",
    "EvaluationConfig",
    "RAGConfig",
]

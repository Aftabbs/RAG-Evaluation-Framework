"""
RAG Evaluator - Main orchestrator for comprehensive RAG system evaluation

This module provides end-to-end evaluation of RAG systems by combining:
- Retrieval quality metrics
- Generation quality metrics
- Overall system performance assessment
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
import json

from src.retrieval_metrics import (
    RetrievalMetricsCalculator,
    RetrievalResult,
    RetrievalGroundTruth,
    ContextRelevanceEvaluator
)
from src.generation_metrics import (
    GenerationMetricsCalculator,
    GenerationResult
)
from src.config import get_config


@dataclass
class RAGTestCase:
    """Test case for RAG evaluation"""
    query: str
    ground_truth_answer: Optional[str] = None
    relevant_doc_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RAGSystemResult:
    """Result from a complete RAG system execution"""
    query: str
    retrieved_doc_ids: List[str]
    retrieved_contexts: List[str]
    generated_answer: str
    retrieval_scores: Optional[List[float]] = None


class RAGEvaluator:
    """
    Comprehensive RAG System Evaluator

    This class orchestrates end-to-end evaluation of RAG systems by:
    1. Evaluating retrieval quality
    2. Evaluating generation quality
    3. Providing overall system assessment

    Example:
        evaluator = RAGEvaluator()
        results = evaluator.evaluate(
            rag_system=my_rag_pipeline,
            test_cases=test_data
        )
    """

    def __init__(
        self,
        config=None,
        k_values: List[int] = [1, 3, 5, 10],
        embedding_model=None
    ):
        """
        Initialize RAG evaluator

        Args:
            config: Configuration object
            k_values: List of K values for retrieval metrics
            embedding_model: Model for embedding-based metrics
        """
        self.config = config or get_config()
        self.retrieval_metrics = RetrievalMetricsCalculator(k_values=k_values)
        self.generation_metrics = GenerationMetricsCalculator(config=self.config)

        # Initialize embedding model for context relevance
        if embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                embedding_model = SentenceTransformer(self.config.embedding_model)
            except Exception as e:
                print(f"Warning: Could not load embedding model: {e}")
                embedding_model = None

        self.context_relevance_evaluator = ContextRelevanceEvaluator(embedding_model)

    def evaluate(
        self,
        rag_system: Callable,
        test_cases: List[RAGTestCase],
        evaluate_retrieval: bool = True,
        evaluate_generation: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a RAG system end-to-end

        Args:
            rag_system: Function that takes a query and returns RAGSystemResult
            test_cases: List of test cases to evaluate
            evaluate_retrieval: Whether to evaluate retrieval quality
            evaluate_generation: Whether to evaluate generation quality
            verbose: Whether to print progress

        Returns:
            Comprehensive evaluation results
        """
        if verbose:
            print(f"Starting evaluation of {len(test_cases)} test cases...")

        all_results = []
        retrieval_results = []
        generation_results = []

        for i, test_case in enumerate(test_cases):
            if verbose and (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_cases)} queries...")

            try:
                # Run RAG system
                rag_result = rag_system(test_case.query)

                # Store for detailed analysis
                all_results.append({
                    "test_case": test_case,
                    "rag_result": rag_result
                })

                # Prepare retrieval evaluation
                if evaluate_retrieval and test_case.relevant_doc_ids:
                    retrieval_result = RetrievalResult(
                        query=test_case.query,
                        retrieved_doc_ids=rag_result.retrieved_doc_ids,
                        retrieved_scores=rag_result.retrieval_scores,
                        retrieved_texts=rag_result.retrieved_contexts
                    )
                    retrieval_results.append(retrieval_result)

                # Prepare generation evaluation
                if evaluate_generation:
                    generation_result = GenerationResult(
                        query=test_case.query,
                        generated_answer=rag_result.generated_answer,
                        retrieved_contexts=rag_result.retrieved_contexts,
                        ground_truth_answer=test_case.ground_truth_answer
                    )
                    generation_results.append(generation_result)

            except Exception as e:
                print(f"Error processing query '{test_case.query}': {e}")
                continue

        # Evaluate retrieval
        retrieval_metrics = {}
        if evaluate_retrieval and retrieval_results:
            if verbose:
                print("\nEvaluating retrieval quality...")

            ground_truths = [
                RetrievalGroundTruth(
                    query=tc.query,
                    relevant_doc_ids=tc.relevant_doc_ids
                )
                for tc in test_cases if tc.relevant_doc_ids
            ]

            retrieval_metrics = self.retrieval_metrics.evaluate_batch(
                retrieval_results,
                ground_truths
            )

            # Add context relevance scores
            context_relevance_scores = []
            for result in all_results:
                try:
                    relevance = self.context_relevance_evaluator.evaluate_context_relevance(
                        result["test_case"].query,
                        result["rag_result"].retrieved_contexts
                    )
                    context_relevance_scores.append(relevance["mean_similarity"])
                except:
                    pass

            if context_relevance_scores:
                retrieval_metrics["context_relevance_mean"] = sum(context_relevance_scores) / len(context_relevance_scores)

        # Evaluate generation
        generation_metrics = {}
        if evaluate_generation and generation_results:
            if verbose:
                print("Evaluating generation quality...")

            generation_metrics = self.generation_metrics.evaluate_batch(
                generation_results
            )

        # Compile final results
        evaluation_results = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_test_cases": len(test_cases),
                "num_successful": len(all_results),
                "config": {
                    "model": self.config.groq_model,
                    "embedding_model": self.config.embedding_model,
                    "k_values": self.retrieval_metrics.k_values
                }
            },
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "overall_score": self._calculate_overall_score(retrieval_metrics, generation_metrics)
        }

        if verbose:
            print("\nEvaluation complete!")
            self.print_summary(evaluation_results)

        return evaluation_results

    def evaluate_retrieval_only(
        self,
        retrieval_results: List[RetrievalResult],
        ground_truths: List[RetrievalGroundTruth]
    ) -> Dict[str, Any]:
        """
        Evaluate only the retrieval component

        Args:
            retrieval_results: List of retrieval results
            ground_truths: List of ground truth data

        Returns:
            Retrieval metrics
        """
        return self.retrieval_metrics.evaluate_batch(retrieval_results, ground_truths)

    def evaluate_generation_only(
        self,
        generation_results: List[GenerationResult]
    ) -> Dict[str, Any]:
        """
        Evaluate only the generation component

        Args:
            generation_results: List of generation results

        Returns:
            Generation metrics
        """
        return self.generation_metrics.evaluate_batch(generation_results)

    def _calculate_overall_score(
        self,
        retrieval_metrics: Dict[str, Any],
        generation_metrics: Dict[str, Any]
    ) -> float:
        """
        Calculate overall RAG system score

        Args:
            retrieval_metrics: Retrieval evaluation results
            generation_metrics: Generation evaluation results

        Returns:
            Overall score (0-1)
        """
        scores = []
        weights = []

        # Retrieval score
        if "ndcg@5_mean" in retrieval_metrics:
            scores.append(retrieval_metrics["ndcg@5_mean"])
            weights.append(0.3)

        if "precision@5_mean" in retrieval_metrics:
            scores.append(retrieval_metrics["precision@5_mean"])
            weights.append(0.2)

        # Generation score
        if "faithfulness_mean" in generation_metrics:
            scores.append(generation_metrics["faithfulness_mean"])
            weights.append(0.25)

        if "answer_relevance_mean" in generation_metrics:
            scores.append(generation_metrics["answer_relevance_mean"])
            weights.append(0.25)

        if not scores:
            return 0.0

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return weighted_score

    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("RAG EVALUATION SUMMARY")
        print("="*60)

        metadata = results.get("evaluation_metadata", {})
        print(f"\nEvaluation Date: {metadata.get('timestamp', 'N/A')}")
        print(f"Test Cases: {metadata.get('num_test_cases', 0)}")
        print(f"Successful: {metadata.get('num_successful', 0)}")

        print("\n" + "-"*60)
        print("RETRIEVAL METRICS")
        print("-"*60)
        retrieval = results.get("retrieval_metrics", {})
        if retrieval:
            for metric in ["precision@5_mean", "recall@5_mean", "ndcg@5_mean", "mrr_mean"]:
                if metric in retrieval:
                    print(f"{metric}: {retrieval[metric]:.4f}")
        else:
            print("No retrieval metrics available")

        print("\n" + "-"*60)
        print("GENERATION METRICS")
        print("-"*60)
        generation = results.get("generation_metrics", {})
        if generation:
            for metric in ["faithfulness_mean", "answer_relevance_mean", "context_utilization_mean", "overall_quality_mean"]:
                if metric in generation:
                    print(f"{metric}: {generation[metric]:.4f}")
        else:
            print("No generation metrics available")

        print("\n" + "-"*60)
        print(f"OVERALL SCORE: {results.get('overall_score', 0.0):.4f}")
        print("-"*60 + "\n")

    def export_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        format: str = "json"
    ):
        """
        Export evaluation results to file

        Args:
            results: Evaluation results
            output_path: Path to save results
            format: Export format ("json" or "csv")
        """
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results exported to {output_path}")

        elif format == "csv":
            # Flatten metrics for CSV export
            rows = []

            # Add retrieval metrics
            retrieval = results.get("retrieval_metrics", {})
            if "detailed_results" in retrieval:
                for detail in retrieval["detailed_results"]:
                    row = {
                        "query": detail["query"],
                        "type": "retrieval",
                    }
                    row.update(detail["metrics"])
                    rows.append(row)

            # Add generation metrics
            generation = results.get("generation_metrics", {})
            if "detailed_results" in generation:
                for detail in generation["detailed_results"]:
                    row = {
                        "query": detail["query"],
                        "type": "generation",
                    }
                    # Flatten nested metrics
                    for key, value in detail["metrics"].items():
                        if isinstance(value, dict) and "score" in value:
                            row[f"{key}_score"] = value["score"]
                        elif isinstance(value, (int, float)):
                            row[key] = value
                    rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            print(f"Results exported to {output_path}")

        else:
            raise ValueError(f"Unsupported format: {format}")

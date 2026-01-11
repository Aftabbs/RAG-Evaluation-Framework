"""
Dataset utilities for RAG evaluation

Provides tools for loading, managing, and creating benchmark datasets
for RAG system evaluation.
"""
from typing import List, Dict, Any, Optional
import json
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from src.evaluator import RAGTestCase


class DatasetLoader:
    """Load and manage evaluation datasets"""

    @staticmethod
    def load_from_json(file_path: str) -> List[RAGTestCase]:
        """
        Load test cases from JSON file

        Expected format:
        [
            {
                "query": "What is...",
                "ground_truth_answer": "The answer is...",
                "relevant_doc_ids": ["doc1", "doc2"],
                "metadata": {...}
            },
            ...
        ]

        Args:
            file_path: Path to JSON file

        Returns:
            List of RAGTestCase objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_cases = []
        for item in data:
            test_case = RAGTestCase(
                query=item["query"],
                ground_truth_answer=item.get("ground_truth_answer"),
                relevant_doc_ids=item.get("relevant_doc_ids"),
                metadata=item.get("metadata")
            )
            test_cases.append(test_case)

        return test_cases

    @staticmethod
    def load_from_csv(file_path: str) -> List[RAGTestCase]:
        """
        Load test cases from CSV file

        Expected columns: query, ground_truth_answer, relevant_doc_ids

        Args:
            file_path: Path to CSV file

        Returns:
            List of RAGTestCase objects
        """
        test_cases = []

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse relevant_doc_ids if present
                relevant_doc_ids = None
                if "relevant_doc_ids" in row and row["relevant_doc_ids"]:
                    relevant_doc_ids = row["relevant_doc_ids"].split("|")

                test_case = RAGTestCase(
                    query=row["query"],
                    ground_truth_answer=row.get("ground_truth_answer"),
                    relevant_doc_ids=relevant_doc_ids
                )
                test_cases.append(test_case)

        return test_cases

    @staticmethod
    def save_to_json(test_cases: List[RAGTestCase], file_path: str):
        """Save test cases to JSON file"""
        data = []
        for tc in test_cases:
            data.append({
                "query": tc.query,
                "ground_truth_answer": tc.ground_truth_answer,
                "relevant_doc_ids": tc.relevant_doc_ids,
                "metadata": tc.metadata
            })

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def save_to_csv(test_cases: List[RAGTestCase], file_path: str):
        """Save test cases to CSV file"""
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["query", "ground_truth_answer", "relevant_doc_ids"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for tc in test_cases:
                writer.writerow({
                    "query": tc.query,
                    "ground_truth_answer": tc.ground_truth_answer,
                    "relevant_doc_ids": "|".join(tc.relevant_doc_ids) if tc.relevant_doc_ids else ""
                })


class BenchmarkDatasets:
    """
    Pre-built benchmark datasets for RAG evaluation

    Provides industry-standard datasets and synthetic datasets
    for testing RAG systems.
    """

    @staticmethod
    def create_sample_qa_dataset() -> List[RAGTestCase]:
        """Create a sample Q&A dataset for testing"""
        return [
            RAGTestCase(
                query="What is machine learning?",
                ground_truth_answer="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
                relevant_doc_ids=["ml_basics_001", "ai_overview_003"],
                metadata={"category": "fundamentals", "difficulty": "easy"}
            ),
            RAGTestCase(
                query="Explain the difference between supervised and unsupervised learning",
                ground_truth_answer="Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data without predefined outputs.",
                relevant_doc_ids=["ml_types_002", "learning_paradigms_005"],
                metadata={"category": "fundamentals", "difficulty": "medium"}
            ),
            RAGTestCase(
                query="What are the key components of a transformer architecture?",
                ground_truth_answer="The key components of a transformer are self-attention mechanisms, multi-head attention, position encodings, feed-forward networks, and layer normalization.",
                relevant_doc_ids=["transformers_101", "attention_mechanisms_007"],
                metadata={"category": "deep_learning", "difficulty": "hard"}
            ),
            RAGTestCase(
                query="How does gradient descent work in neural networks?",
                ground_truth_answer="Gradient descent is an optimization algorithm that iteratively adjusts model parameters in the direction opposite to the gradient of the loss function to minimize error.",
                relevant_doc_ids=["optimization_004", "neural_nets_basics_002"],
                metadata={"category": "optimization", "difficulty": "medium"}
            ),
            RAGTestCase(
                query="What is the purpose of regularization in machine learning?",
                ground_truth_answer="Regularization prevents overfitting by adding a penalty term to the loss function, encouraging the model to learn simpler patterns that generalize better to unseen data.",
                relevant_doc_ids=["regularization_techniques_006", "overfitting_solutions_003"],
                metadata={"category": "model_training", "difficulty": "medium"}
            )
        ]

    @staticmethod
    def create_multi_hop_qa_dataset() -> List[RAGTestCase]:
        """
        Create dataset requiring multi-hop reasoning

        These questions require synthesizing information from multiple sources
        """
        return [
            RAGTestCase(
                query="If transformers use self-attention and BERT is a transformer, what mechanism does BERT use?",
                ground_truth_answer="BERT uses self-attention mechanisms since it is based on the transformer architecture.",
                relevant_doc_ids=["transformers_101", "bert_architecture_008", "attention_mechanisms_007"],
                metadata={"category": "reasoning", "difficulty": "hard", "hops": 2}
            ),
            RAGTestCase(
                query="What optimization technique is commonly used to train models that prevent overfitting?",
                ground_truth_answer="Gradient descent combined with regularization techniques like L1/L2 regularization, dropout, or early stopping.",
                relevant_doc_ids=["optimization_004", "regularization_techniques_006", "training_best_practices_009"],
                metadata={"category": "reasoning", "difficulty": "hard", "hops": 2}
            )
        ]

    @staticmethod
    def create_adversarial_dataset() -> List[RAGTestCase]:
        """
        Create adversarial examples to test robustness

        These test cases have misleading or contradictory information
        """
        return [
            RAGTestCase(
                query="Is deep learning the same as machine learning?",
                ground_truth_answer="No, deep learning is a subset of machine learning that uses neural networks with multiple layers.",
                relevant_doc_ids=["ml_basics_001", "deep_learning_intro_010"],
                metadata={"category": "adversarial", "type": "clarification"}
            ),
            RAGTestCase(
                query="Can you train a model without data?",
                ground_truth_answer="No, you cannot train a machine learning model without data. Data is essential for learning patterns.",
                relevant_doc_ids=["ml_basics_001", "data_requirements_011"],
                metadata={"category": "adversarial", "type": "impossible_scenario"}
            )
        ]


class DatasetAnalyzer:
    """Analyze dataset characteristics"""

    @staticmethod
    def analyze_dataset(test_cases: List[RAGTestCase]) -> Dict[str, Any]:
        """
        Analyze characteristics of a test dataset

        Args:
            test_cases: List of test cases

        Returns:
            Analysis report
        """
        analysis = {
            "total_cases": len(test_cases),
            "with_ground_truth": sum(1 for tc in test_cases if tc.ground_truth_answer),
            "with_relevant_docs": sum(1 for tc in test_cases if tc.relevant_doc_ids),
            "avg_query_length": sum(len(tc.query.split()) for tc in test_cases) / len(test_cases) if test_cases else 0,
            "avg_answer_length": 0,
            "avg_relevant_docs": 0,
        }

        # Calculate average answer length
        answers = [tc.ground_truth_answer for tc in test_cases if tc.ground_truth_answer]
        if answers:
            analysis["avg_answer_length"] = sum(len(a.split()) for a in answers) / len(answers)

        # Calculate average number of relevant docs
        relevant_docs = [tc.relevant_doc_ids for tc in test_cases if tc.relevant_doc_ids]
        if relevant_docs:
            analysis["avg_relevant_docs"] = sum(len(docs) for docs in relevant_docs) / len(relevant_docs)

        # Category distribution
        categories = {}
        for tc in test_cases:
            if tc.metadata and "category" in tc.metadata:
                cat = tc.metadata["category"]
                categories[cat] = categories.get(cat, 0) + 1

        analysis["category_distribution"] = categories

        return analysis

    @staticmethod
    def print_analysis(analysis: Dict[str, Any]):
        """Print dataset analysis"""
        print("\n" + "="*60)
        print("DATASET ANALYSIS")
        print("="*60)
        print(f"Total test cases: {analysis['total_cases']}")
        print(f"With ground truth answers: {analysis['with_ground_truth']}")
        print(f"With relevant doc IDs: {analysis['with_relevant_docs']}")
        print(f"Avg query length: {analysis['avg_query_length']:.1f} words")
        print(f"Avg answer length: {analysis['avg_answer_length']:.1f} words")
        print(f"Avg relevant docs per query: {analysis['avg_relevant_docs']:.1f}")

        if analysis.get("category_distribution"):
            print("\nCategory distribution:")
            for cat, count in analysis["category_distribution"].items():
                print(f"  {cat}: {count}")

        print("="*60 + "\n")

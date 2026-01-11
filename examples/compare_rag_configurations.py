"""
Compare different RAG configurations

This example demonstrates how to use the evaluation framework to:
1. Test different chunk sizes
2. Compare different numbers of retrieved documents (top_k)
3. Identify the optimal configuration for your use case
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain.schema import Document
import pandas as pd

from src.evaluator import RAGEvaluator, RAGSystemResult
from src.dataset_utils import BenchmarkDatasets
from examples.evaluate_rag_system import SimpleRAGSystem, create_sample_knowledge_base

load_dotenv()


def compare_configurations():
    """Compare different RAG configurations"""

    print("\n" + "="*70)
    print("RAG CONFIGURATION COMPARISON")
    print("="*70 + "\n")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not found")
        return

    # Load test data
    documents = create_sample_knowledge_base()
    datasets = BenchmarkDatasets()
    test_cases = datasets.create_sample_qa_dataset()

    # Define configurations to test
    configurations = [
        {"name": "Small chunks, Top-3", "chunk_size": 256, "chunk_overlap": 25, "top_k": 3},
        {"name": "Medium chunks, Top-3", "chunk_size": 512, "chunk_overlap": 50, "top_k": 3},
        {"name": "Medium chunks, Top-5", "chunk_size": 512, "chunk_overlap": 50, "top_k": 5},
        {"name": "Large chunks, Top-5", "chunk_size": 1024, "chunk_overlap": 100, "top_k": 5},
    ]

    results_comparison = []

    for config in configurations:
        print(f"\nTesting: {config['name']}")
        print(f"  Chunk size: {config['chunk_size']}")
        print(f"  Chunk overlap: {config['chunk_overlap']}")
        print(f"  Top-K: {config['top_k']}")
        print()

        # Initialize RAG system with this configuration
        rag_system = SimpleRAGSystem(
            documents=documents,
            groq_api_key=groq_api_key,
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap'],
            top_k=config['top_k']
        )

        # Create evaluator
        evaluator = RAGEvaluator(k_values=[1, 3, 5])

        # Run evaluation
        results = evaluator.evaluate(
            rag_system=lambda q: rag_system.query(q),
            test_cases=test_cases,
            evaluate_retrieval=True,
            evaluate_generation=True,
            verbose=False
        )

        # Store results
        result_summary = {
            "Configuration": config['name'],
            "Chunk Size": config['chunk_size'],
            "Top-K": config['top_k'],
            "Overall Score": results['overall_score'],
        }

        # Add key metrics
        retrieval = results.get('retrieval_metrics', {})
        generation = results.get('generation_metrics', {})

        result_summary["Precision@5"] = retrieval.get('precision@5_mean', 0)
        result_summary["Recall@5"] = retrieval.get('recall@5_mean', 0)
        result_summary["NDCG@5"] = retrieval.get('ndcg@5_mean', 0)
        result_summary["Faithfulness"] = generation.get('faithfulness_mean', 0)
        result_summary["Answer Relevance"] = generation.get('answer_relevance_mean', 0)

        results_comparison.append(result_summary)

        print(f"  Overall Score: {results['overall_score']:.4f}")
        print(f"  Faithfulness: {result_summary['Faithfulness']:.4f}")
        print(f"  Answer Relevance: {result_summary['Answer Relevance']:.4f}")

    # Create comparison table
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70 + "\n")

    df = pd.DataFrame(results_comparison)
    df = df.sort_values('Overall Score', ascending=False)

    print(df.to_string(index=False))

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "configuration_comparison.csv", index=False)

    print(f"\nResults saved to {output_dir}/configuration_comparison.csv")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    best_config = df.iloc[0]
    print(f"\nBest Overall Configuration: {best_config['Configuration']}")
    print(f"  - Overall Score: {best_config['Overall Score']:.4f}")
    print(f"  - Chunk Size: {int(best_config['Chunk Size'])}")
    print(f"  - Top-K: {int(best_config['Top-K'])}")

    best_faithfulness = df.loc[df['Faithfulness'].idxmax()]
    print(f"\nBest for Faithfulness: {best_faithfulness['Configuration']}")
    print(f"  - Faithfulness Score: {best_faithfulness['Faithfulness']:.4f}")

    best_retrieval = df.loc[df['NDCG@5'].idxmax()]
    print(f"\nBest for Retrieval: {best_retrieval['Configuration']}")
    print(f"  - NDCG@5: {best_retrieval['NDCG@5']:.4f}")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    compare_configurations()

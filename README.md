# RAG Evaluation Framework

<img width="1161" height="565" alt="image" src="https://github.com/user-attachments/assets/e0f0b3d0-d8af-40a7-9f68-29502c1fbce7" />

A production-ready framework for comprehensively evaluating Retrieval-Augmented Generation (RAG) systems using industry-standard metrics and LLM-based evaluation with Groq and LangChain.

## Overview

Evaluating RAG systems is challenging because performance depends on two distinct components:
1. **Retrieval Quality**: How well the system finds relevant information
2. **Generation Quality**: How well the LLM synthesizes that information into accurate, relevant answers

This framework provides rigorous evaluation of both components with metrics used in production AI systems.

## Features

### Retrieval Metrics
- **Precision@K**: Proportion of retrieved docs that are relevant
- **Recall@K**: Proportion of relevant docs that are retrieved
- **Mean Reciprocal Rank (MRR)**: Position of first relevant document
- **NDCG (Normalized Discounted Cumulative Gain)**: Ranking quality
- **Hit Rate**: Whether any relevant document was retrieved
- **Mean Average Precision (MAP)**: Overall retrieval quality

### Generation Metrics (LLM-Based)
- **Faithfulness**: Detects hallucinations by checking if answers are grounded in contexts
- **Answer Relevance**: Measures if the answer addresses the question
- **Context Utilization**: Evaluates how well the answer uses retrieved information
- **Semantic Similarity**: Compares generated answer to ground truth
- **ROUGE Scores**: Traditional overlap-based metrics

### Key Capabilities
- ✅ End-to-end RAG pipeline evaluation
- ✅ LangChain and LangGraph integration
- ✅ Groq API for fast, cost-effective LLM evaluation
- ✅ Batch evaluation with progress tracking
- ✅ Configuration comparison (chunk size, top-k, etc.)
- ✅ Export results to JSON/CSV
- ✅ Pre-built benchmark datasets
- ✅ Extensible architecture for custom metrics

## Installation

```bash
# Clone the repository
cd rag-evaluation-framework

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Dependencies

- LangChain & LangGraph for RAG pipelines
- Groq for LLM-based evaluation
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- ROUGE, BERTScore for traditional metrics

## Quick Start

### 1. Basic Evaluation

```python
from src.evaluator import RAGEvaluator, RAGTestCase, RAGSystemResult
from dotenv import load_dotenv

load_dotenv()

# Define your test cases
test_cases = [
    RAGTestCase(
        query="What is machine learning?",
        ground_truth_answer="Machine learning is...",
        relevant_doc_ids=["doc1", "doc3"]
    ),
    # ... more test cases
]

# Create evaluator
evaluator = RAGEvaluator(k_values=[1, 3, 5])

# Your RAG system wrapper
def my_rag_system(query: str) -> RAGSystemResult:
    # Your RAG logic here
    return RAGSystemResult(
        query=query,
        retrieved_doc_ids=[...],
        retrieved_contexts=[...],
        generated_answer="..."
    )

# Run evaluation
results = evaluator.evaluate(
    rag_system=my_rag_system,
    test_cases=test_cases,
    evaluate_retrieval=True,
    evaluate_generation=True
)

# View results
evaluator.print_summary(results)
```

### 2. Full Example with LangChain

See `examples/evaluate_rag_system.py` for a complete working example:

```bash
python examples/evaluate_rag_system.py
```

This example demonstrates:
- Building a RAG system with LangChain and Groq
- Using ChromaDB for vector storage
- Loading benchmark datasets
- Running comprehensive evaluation
- Exporting results

### 3. Compare Configurations

Compare different RAG configurations to find the optimal setup:

```bash
python examples/compare_rag_configurations.py
```

This compares:
- Different chunk sizes (256, 512, 1024)
- Different top-k values (3, 5, 10)
- Different chunk overlaps

## Project Structure

```
rag-evaluation-framework/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   ├── utils.py                 # Utility functions
│   ├── retrieval_metrics.py     # Retrieval evaluation metrics
│   ├── generation_metrics.py    # Generation evaluation metrics
│   ├── evaluator.py             # Main evaluation orchestrator
│   └── dataset_utils.py         # Dataset loading and management
├── examples/
│   ├── evaluate_rag_system.py         # Full RAG evaluation example
│   └── compare_rag_configurations.py  # Configuration comparison
├── tests/
│   └── test_metrics.py          # Unit tests
├── results/                     # Evaluation results (auto-generated)
├── requirements.txt             # Dependencies
├── .env.example                 # Environment variables template
└── README.md                    # This file
```

## Detailed Usage

### Retrieval-Only Evaluation

If you only want to evaluate retrieval quality:

```python
from src.retrieval_metrics import (
    RetrievalMetricsCalculator,
    RetrievalResult,
    RetrievalGroundTruth
)

calculator = RetrievalMetricsCalculator(k_values=[1, 3, 5, 10])

# Your retrieval results
results = [
    RetrievalResult(
        query="What is AI?",
        retrieved_doc_ids=["doc1", "doc2", "doc3"]
    ),
    # ... more results
]

# Ground truth
ground_truths = [
    RetrievalGroundTruth(
        query="What is AI?",
        relevant_doc_ids=["doc2", "doc5"]
    ),
    # ... more ground truths
]

# Evaluate
metrics = calculator.evaluate_batch(results, ground_truths)
print(metrics)
```

### Generation-Only Evaluation

If you only want to evaluate generation quality:

```python
from src.generation_metrics import (
    GenerationMetricsCalculator,
    GenerationResult
)

calculator = GenerationMetricsCalculator()

results = [
    GenerationResult(
        query="What is AI?",
        generated_answer="AI is...",
        retrieved_contexts=["Context 1...", "Context 2..."],
        ground_truth_answer="Ground truth..."
    ),
    # ... more results
]

metrics = calculator.evaluate_batch(results)
print(metrics)
```

### Using Benchmark Datasets

Pre-built datasets for quick testing:

```python
from src.dataset_utils import BenchmarkDatasets, DatasetAnalyzer

datasets = BenchmarkDatasets()

# Load sample QA dataset
test_cases = datasets.create_sample_qa_dataset()

# Load multi-hop reasoning dataset
multi_hop = datasets.create_multi_hop_qa_dataset()

# Load adversarial dataset
adversarial = datasets.create_adversarial_dataset()

# Analyze dataset characteristics
analyzer = DatasetAnalyzer()
analysis = analyzer.analyze_dataset(test_cases)
analyzer.print_analysis(analysis)
```

### Loading Custom Datasets

Load your own test data:

```python
from src.dataset_utils import DatasetLoader

# From JSON
test_cases = DatasetLoader.load_from_json("my_test_data.json")

# From CSV
test_cases = DatasetLoader.load_from_csv("my_test_data.csv")

# Save datasets
DatasetLoader.save_to_json(test_cases, "output.json")
```

## Evaluation Metrics Explained

### Retrieval Metrics

**Precision@K**: Of the K documents retrieved, how many are relevant?
- Range: 0-1 (higher is better)
- Use case: When you care about minimizing irrelevant results

**Recall@K**: Of all relevant documents, how many are in the top K?
- Range: 0-1 (higher is better)
- Use case: When you need to find all relevant information

**NDCG@K**: Quality of ranking (rewards relevant docs at top positions)
- Range: 0-1 (higher is better)
- Use case: When ranking order matters

**MRR**: Reciprocal of the rank of the first relevant document
- Range: 0-1 (higher is better)
- Use case: When finding at least one relevant doc quickly matters

### Generation Metrics

**Faithfulness**: Does the answer only contain information from the contexts?
- Range: 0-1 (higher is better)
- Use case: Detecting hallucinations
- Evaluation: LLM judges if claims are supported by contexts

**Answer Relevance**: Does the answer address the question?
- Range: 0-1 (higher is better)
- Use case: Ensuring on-topic responses
- Evaluation: LLM judges relevance to query

**Context Utilization**: How well does the answer use retrieved information?
- Range: 0-1 (higher is better)
- Use case: Ensuring contexts are being used effectively
- Evaluation: LLM assesses information synthesis

## Configuration

Edit `.env` file:

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional - customize models
GROQ_MODEL=mixtral-8x7b-32768  # or llama3-70b-8192
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional - evaluation settings
BATCH_SIZE=10
MAX_CONCURRENT_REQUESTS=5
```

## Running Tests

```bash
# Run all tests
python tests/test_metrics.py

# Or with pytest
pytest tests/
```

## Use Cases

### 1. Model Selection
Compare different LLMs for your RAG system:
- GPT-4 vs Claude vs Mixtral
- Different model sizes (7B vs 70B)

### 2. Configuration Optimization
Find optimal settings:
- Chunk size (128, 256, 512, 1024 tokens)
- Chunk overlap (0%, 10%, 20%)
- Top-K retrieval (3, 5, 10 documents)

### 3. Continuous Monitoring
Track RAG performance over time:
- Monitor metrics across versions
- Detect degradation
- A/B test improvements

### 4. Debugging
Identify issues:
- Low faithfulness → hallucination problems
- Low precision → retrieval returning irrelevant docs
- Low context utilization → LLM not using contexts well

## Best Practices

1. **Always evaluate both retrieval and generation**: Problems in either component affect overall performance

2. **Use multiple K values**: Different applications need different retrieval depths

3. **Include diverse test cases**: Cover different question types, difficulties, and edge cases

4. **Track over time**: Create a baseline and monitor changes

5. **Use ground truth when possible**: Enables more reliable metrics

6. **Consider faithfulness critical**: Hallucinations are the biggest risk in RAG

## Contributing

Contributions welcome! Areas for expansion:
- Additional metrics (e.g., coherence, fluency)
- More benchmark datasets
- Integration with other vector stores
- Support for more LLM providers

## License

MIT License - feel free to use in your projects

## Citation

If you use this framework in your research or production systems, please cite:

```
RAG Evaluation Framework (2024)
https://github.com/yourusername/rag-evaluation-framework
```

## Acknowledgments

Built with:
- LangChain for RAG orchestration
- Groq for fast LLM inference
- Sentence Transformers for embeddings
- ChromaDB for vector storage

## Support

- Issues: GitHub Issues
- Questions: Discussions tab
- Documentation: This README and code comments

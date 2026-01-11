# Getting Started with RAG Evaluation Framework

This guide will walk you through setting up and running your first RAG evaluation in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- Groq API key (get free at [groq.com](https://groq.com))
- Basic familiarity with RAG systems

## Step 1: Installation (2 minutes)

```bash
# Navigate to the project directory
cd rag-evaluation-framework

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configuration (1 minute)

Create a `.env` file in the project root:

```bash
# Copy the example
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=your_actual_groq_api_key_here
```

Get your free Groq API key at: https://console.groq.com/keys

## Step 3: Run Your First Evaluation (2 minutes)

Run the complete example:

```bash
python examples/evaluate_rag_system.py
```

This will:
1. Create a sample knowledge base about AI/ML
2. Build a RAG system using LangChain and Groq
3. Run comprehensive evaluation on 5 test queries
4. Display detailed metrics
5. Save results to `results/` directory

## Understanding the Output

You'll see metrics like:

### Retrieval Metrics
- `precision@5_mean: 0.8000` → 80% of retrieved docs are relevant
- `recall@5_mean: 0.6667` → Retrieved 67% of all relevant docs
- `ndcg@5_mean: 0.7500` → Good ranking quality
- `mrr_mean: 0.9000` → First relevant doc appears early

### Generation Metrics
- `faithfulness_mean: 0.8500` → 85% of answer is grounded in contexts
- `answer_relevance_mean: 0.9000` → 90% relevant to the question
- `context_utilization_mean: 0.7500` → Good use of retrieved info

### Overall Score
- `overall_score: 0.8200` → Combined performance score

## Step 4: Compare Configurations (Optional)

See which RAG configuration works best:

```bash
python examples/compare_rag_configurations.py
```

This compares different chunk sizes and top-k values.

## Next Steps

### Use Your Own Data

Create a test dataset JSON file:

```json
[
  {
    "query": "Your question here",
    "ground_truth_answer": "Expected answer",
    "relevant_doc_ids": ["doc1", "doc2"]
  }
]
```

Load and use it:

```python
from src.dataset_utils import DatasetLoader

test_cases = DatasetLoader.load_from_json("your_dataset.json")
```

### Integrate Your RAG System

Wrap your existing RAG system:

```python
from src.evaluator import RAGSystemResult

def your_rag_wrapper(query: str) -> RAGSystemResult:
    # Call your existing RAG system
    docs, answer = your_rag_system.query(query)

    return RAGSystemResult(
        query=query,
        retrieved_doc_ids=[doc.id for doc in docs],
        retrieved_contexts=[doc.content for doc in docs],
        generated_answer=answer
    )
```

Then evaluate:

```python
from src.evaluator import RAGEvaluator

evaluator = RAGEvaluator()
results = evaluator.evaluate(
    rag_system=your_rag_wrapper,
    test_cases=test_cases
)
```

### View Detailed Results

Check the `results/` directory for:
- `evaluation_results.json` - Full results with all metrics
- `evaluation_results.csv` - Tabular format for spreadsheets

## Common Issues

### "GROQ_API_KEY not found"
- Make sure you created `.env` file in the project root
- Verify the API key is correct
- Check there are no quotes around the key

### Import Errors
- Ensure you're in the project root directory
- Activate your virtual environment
- Reinstall dependencies: `pip install -r requirements.txt`

### ChromaDB Errors
- ChromaDB may need to be reset: delete any `.chroma` directories
- On Windows, you may need Visual C++ redistributables

## Understanding the Framework

### Key Components

1. **Retrieval Metrics** (`src/retrieval_metrics.py`)
   - Measures how well documents are retrieved
   - Uses established IR metrics

2. **Generation Metrics** (`src/generation_metrics.py`)
   - Uses Groq LLM to evaluate answer quality
   - Detects hallucinations, measures relevance

3. **Evaluator** (`src/evaluator.py`)
   - Orchestrates the full evaluation
   - Combines retrieval and generation metrics

4. **Dataset Utils** (`src/dataset_utils.py`)
   - Load/save test datasets
   - Pre-built benchmark datasets

### Evaluation Pipeline

```
Your RAG System
      ↓
Query → [Retrieval] → Retrieved Docs
              ↓
        [Generation] → Answer
              ↓
      [Evaluation]
              ↓
    Metrics & Scores
```

## Metrics Deep Dive

### When to Focus on Each Metric

**High Faithfulness is Critical When:**
- Building customer-facing applications
- Providing factual information
- Legal, medical, or financial domains

**High Precision is Critical When:**
- Users have limited time
- Irrelevant info is harmful
- Context window is limited

**High Recall is Critical When:**
- Completeness matters
- Missing information is costly
- Research or analysis tasks

## Tips for Better Evaluation

1. **Start Small**: Test on 10-20 queries first
2. **Iterate**: Run eval → identify issues → fix → re-eval
3. **Track Changes**: Save results after each configuration change
4. **Use Multiple Datasets**: Include easy, medium, hard questions
5. **Monitor Over Time**: Create a baseline and track improvements

## Example Workflow

```bash
# 1. Initial evaluation
python examples/evaluate_rag_system.py
# → Overall score: 0.65

# 2. Compare configurations
python examples/compare_rag_configurations.py
# → Find that chunk_size=512, top_k=5 is best

# 3. Update your RAG system with best config

# 4. Re-evaluate
python examples/evaluate_rag_system.py
# → Overall score: 0.82 (improvement!)

# 5. Save baseline
cp results/evaluation_results.json results/baseline_v1.json
```

## Learning Resources

- **LangChain Docs**: https://python.langchain.com/
- **Groq API**: https://console.groq.com/docs
- **RAG Paper**: https://arxiv.org/abs/2005.11401
- **Evaluation Metrics**: Check `README.md` for detailed explanations

## Getting Help

- Read the code comments - they're detailed
- Check `examples/` for working code
- Run tests: `python tests/test_metrics.py`
- Open an issue on GitHub

## Quick Reference

```python
# Minimal evaluation example
from src.evaluator import RAGEvaluator, RAGTestCase, RAGSystemResult

test_cases = [RAGTestCase(query="What is AI?")]
evaluator = RAGEvaluator()

results = evaluator.evaluate(
    rag_system=my_rag_function,
    test_cases=test_cases
)

evaluator.print_summary(results)
```

Ready to start? Run:
```bash
python examples/evaluate_rag_system.py
```

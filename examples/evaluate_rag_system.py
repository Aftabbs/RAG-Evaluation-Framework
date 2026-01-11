"""
Comprehensive example of building and evaluating a RAG system

This example demonstrates:
1. Building a RAG pipeline with LangChain and Groq
2. Loading evaluation datasets
3. Running comprehensive evaluation
4. Analyzing results
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from src.evaluator import RAGEvaluator, RAGTestCase, RAGSystemResult
from src.dataset_utils import BenchmarkDatasets, DatasetAnalyzer

# Load environment variables
load_dotenv()


class SimpleRAGSystem:
    """
    Example RAG system using LangChain and Groq

    This is a production-ready RAG implementation that:
    - Uses HuggingFace embeddings for semantic search
    - Stores vectors in ChromaDB
    - Uses Groq's Mixtral for answer generation
    """

    def __init__(
        self,
        documents: list[Document],
        groq_api_key: str,
        model_name: str = "mixtral-8x7b-32768",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5
    ):
        """
        Initialize RAG system

        Args:
            documents: List of LangChain documents to index
            groq_api_key: Groq API key
            model_name: Groq model to use
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve
        """
        self.top_k = top_k

        # Initialize embeddings
        print("Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Split documents
        print(f"Splitting {len(documents)} documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        print(f"Created {len(splits)} chunks")

        # Create vector store
        print("Creating vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name="rag_eval_demo"
        )

        # Initialize LLM
        self.llm = ChatGroq(
            model=model_name,
            temperature=0.1,
            api_key=groq_api_key
        )

        # Create custom prompt
        template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer or the context doesn't contain enough information, say so clearly. Do not make up information.

Context:
{context}

Question: {question}

Answer: """

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Create retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": top_k}
            ),
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )

    def query(self, question: str) -> RAGSystemResult:
        """
        Query the RAG system

        Args:
            question: User question

        Returns:
            RAGSystemResult with retrieved docs and generated answer
        """
        # Get results from QA chain
        result = self.qa_chain({"query": question})

        # Extract retrieved documents
        retrieved_docs = result.get("source_documents", [])

        # Extract document IDs and contexts
        retrieved_doc_ids = [doc.metadata.get("id", f"doc_{i}") for i, doc in enumerate(retrieved_docs)]
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]

        # Get generated answer
        generated_answer = result.get("result", "")

        return RAGSystemResult(
            query=question,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_contexts=retrieved_contexts,
            generated_answer=generated_answer
        )


def create_sample_knowledge_base() -> list[Document]:
    """Create a sample knowledge base about AI/ML"""

    documents = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
            metadata={"id": "ml_basics_001", "topic": "fundamentals"}
        ),
        Document(
            page_content="There are three main types of machine learning: supervised learning (using labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through interaction with an environment).",
            metadata={"id": "ml_types_002", "topic": "fundamentals"}
        ),
        Document(
            page_content="Deep learning is a subset of machine learning that uses neural networks with multiple layers. These deep neural networks can learn hierarchical representations of data, making them particularly effective for tasks like image recognition and natural language processing.",
            metadata={"id": "deep_learning_intro_010", "topic": "deep_learning"}
        ),
        Document(
            page_content="The transformer architecture revolutionized natural language processing. It uses self-attention mechanisms to process input sequences in parallel, rather than sequentially like RNNs. Key components include multi-head attention, position encodings, and feed-forward networks.",
            metadata={"id": "transformers_101", "topic": "architectures"}
        ),
        Document(
            page_content="BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model that learns contextualized word representations by training on masked language modeling and next sentence prediction tasks.",
            metadata={"id": "bert_architecture_008", "topic": "architectures"}
        ),
        Document(
            page_content="Self-attention mechanisms allow models to weigh the importance of different parts of the input when processing each element. This enables transformers to capture long-range dependencies more effectively than traditional RNNs.",
            metadata={"id": "attention_mechanisms_007", "topic": "architectures"}
        ),
        Document(
            page_content="Gradient descent is an optimization algorithm that iteratively adjusts model parameters to minimize a loss function. It calculates the gradient (derivative) of the loss with respect to parameters and updates them in the opposite direction.",
            metadata={"id": "optimization_004", "topic": "training"}
        ),
        Document(
            page_content="Regularization techniques like L1/L2 regularization, dropout, and early stopping help prevent overfitting by adding constraints or penalties to the model. This encourages learning simpler patterns that generalize better to unseen data.",
            metadata={"id": "regularization_techniques_006", "topic": "training"}
        ),
        Document(
            page_content="Neural networks consist of layers of interconnected nodes (neurons). Each connection has a weight, and each neuron applies an activation function to its inputs. The network learns by adjusting these weights through backpropagation.",
            metadata={"id": "neural_nets_basics_002", "topic": "fundamentals"}
        ),
        Document(
            page_content="Overfitting occurs when a model learns the training data too well, including noise and outliers, resulting in poor generalization to new data. Solutions include regularization, more training data, simpler models, and cross-validation.",
            metadata={"id": "overfitting_solutions_003", "topic": "training"}
        ),
    ]

    return documents


def main():
    """Main evaluation pipeline"""

    print("\n" + "="*70)
    print("RAG SYSTEM EVALUATION DEMO")
    print("="*70 + "\n")

    # Check for API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not found in environment variables")
        print("Please create a .env file with your Groq API key")
        return

    # Step 1: Create knowledge base
    print("Step 1: Creating sample knowledge base...")
    documents = create_sample_knowledge_base()
    print(f"Created knowledge base with {len(documents)} documents\n")

    # Step 2: Initialize RAG system
    print("Step 2: Initializing RAG system with LangChain and Groq...")
    rag_system = SimpleRAGSystem(
        documents=documents,
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768",
        top_k=5
    )
    print("RAG system initialized\n")

    # Step 3: Load evaluation dataset
    print("Step 3: Loading evaluation dataset...")
    datasets = BenchmarkDatasets()
    test_cases = datasets.create_sample_qa_dataset()

    # Analyze dataset
    analyzer = DatasetAnalyzer()
    analysis = analyzer.analyze_dataset(test_cases)
    analyzer.print_analysis(analysis)

    # Step 4: Run evaluation
    print("Step 4: Running comprehensive evaluation...")
    print("This will evaluate both retrieval and generation quality...\n")

    evaluator = RAGEvaluator(
        k_values=[1, 3, 5],  # Evaluate at different K values
    )

    # Wrapper function for evaluation
    def rag_query_wrapper(query: str) -> RAGSystemResult:
        return rag_system.query(query)

    # Run evaluation
    results = evaluator.evaluate(
        rag_system=rag_query_wrapper,
        test_cases=test_cases,
        evaluate_retrieval=True,
        evaluate_generation=True,
        verbose=True
    )

    # Step 5: Export results
    print("\nStep 5: Exporting results...")
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    evaluator.export_results(
        results,
        str(output_dir / "evaluation_results.json"),
        format="json"
    )

    evaluator.export_results(
        results,
        str(output_dir / "evaluation_results.csv"),
        format="csv"
    )

    print(f"Results exported to {output_dir}/\n")

    # Step 6: Show detailed insights
    print("="*70)
    print("DETAILED INSIGHTS")
    print("="*70)

    # Show some example queries and their evaluation
    gen_metrics = results.get("generation_metrics", {})
    if "detailed_results" in gen_metrics:
        print("\nSample Query Analysis:")
        for i, detail in enumerate(gen_metrics["detailed_results"][:2], 1):
            print(f"\n--- Query {i} ---")
            print(f"Q: {detail['query']}")
            print(f"A: {detail['answer'][:200]}...")

            metrics = detail['metrics']
            if 'faithfulness' in metrics:
                print(f"\nFaithfulness: {metrics['faithfulness']['score']:.3f}")
                print(f"Reasoning: {metrics['faithfulness']['reasoning']}")

            if 'answer_relevance' in metrics:
                print(f"\nAnswer Relevance: {metrics['answer_relevance']['score']:.3f}")
                print(f"Reasoning: {metrics['answer_relevance']['reasoning']}")

    print("\n" + "="*70)
    print("Evaluation complete! Check the results directory for full details.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

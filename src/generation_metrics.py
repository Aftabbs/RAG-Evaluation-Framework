"""
Generation Metrics for RAG Evaluation

Implements LLM-based and traditional metrics for evaluating answer generation quality:
- Faithfulness (hallucination detection)
- Answer Relevance
- Context Utilization
- Semantic Similarity
- ROUGE scores
"""
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json

from src.utils import normalize_text, extract_json_from_text
from src.config import get_config


class FaithfulnessScore(BaseModel):
    """Output format for faithfulness evaluation"""
    score: float = Field(description="Faithfulness score between 0 and 1")
    reasoning: str = Field(description="Explanation for the score")
    unsupported_claims: List[str] = Field(default=[], description="Claims not supported by context")


class RelevanceScore(BaseModel):
    """Output format for relevance evaluation"""
    score: float = Field(description="Relevance score between 0 and 1")
    reasoning: str = Field(description="Explanation for the score")


@dataclass
class GenerationResult:
    """Result from answer generation"""
    query: str
    generated_answer: str
    retrieved_contexts: List[str]
    ground_truth_answer: Optional[str] = None


class GenerationMetricsCalculator:
    """
    Calculator for generation quality metrics in RAG systems

    Uses both LLM-based evaluation (via Groq) and traditional NLP metrics
    to provide comprehensive assessment of answer quality.
    """

    def __init__(self, config=None):
        """
        Initialize generation metrics calculator

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or get_config()

        # Initialize Groq LLM for evaluation
        self.llm = ChatGroq(
            model=self.config.groq_model,
            temperature=self.config.evaluation_temperature,
            api_key=self.config.groq_api_key
        )

    def evaluate_single(
        self,
        result: GenerationResult,
        include_traditional_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a single generated answer

        Args:
            result: Generation result to evaluate
            include_traditional_metrics: Whether to include ROUGE, BLEU, etc.

        Returns:
            Dictionary of metric scores
        """
        metrics = {}

        # LLM-based metrics
        metrics["faithfulness"] = self.evaluate_faithfulness(
            result.generated_answer,
            result.retrieved_contexts
        )

        metrics["answer_relevance"] = self.evaluate_answer_relevance(
            result.query,
            result.generated_answer
        )

        metrics["context_utilization"] = self.evaluate_context_utilization(
            result.generated_answer,
            result.retrieved_contexts
        )

        # Traditional metrics (if ground truth available)
        if include_traditional_metrics and result.ground_truth_answer:
            traditional = self._calculate_traditional_metrics(
                result.generated_answer,
                result.ground_truth_answer
            )
            metrics.update(traditional)

        # Overall quality score
        metrics["overall_quality"] = self._calculate_overall_quality(metrics)

        return metrics

    def evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate if the answer is faithful to the retrieved contexts

        Faithfulness measures whether the generated answer contains only
        information that can be derived from the provided contexts.
        This is critical for detecting hallucinations.

        Args:
            answer: Generated answer
            contexts: Retrieved contexts

        Returns:
            Faithfulness score and details
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator assessing the faithfulness of AI-generated answers.

Your task: Determine if the answer contains ONLY information that can be derived from the provided contexts.

Scoring Guide:
- 1.0: All claims in the answer are fully supported by the contexts
- 0.7-0.9: Most claims supported, minor unsupported details
- 0.4-0.6: Mix of supported and unsupported claims
- 0.1-0.3: Most claims not supported by contexts
- 0.0: Answer completely contradicts or ignores contexts

Identify any specific claims in the answer that are NOT supported by the contexts."""),
            ("user", """Contexts:
{contexts}

Answer to evaluate:
{answer}

Provide your evaluation in JSON format:
{{
    "score": <float between 0 and 1>,
    "reasoning": "<explanation>",
    "unsupported_claims": [<list of unsupported claims, if any>]
}}""")
        ])

        try:
            contexts_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])

            chain = prompt | self.llm
            response = chain.invoke({
                "contexts": contexts_text,
                "answer": answer
            })

            # Parse response
            result = extract_json_from_text(response.content)
            if result:
                return {
                    "score": float(result.get("score", 0.0)),
                    "reasoning": result.get("reasoning", ""),
                    "unsupported_claims": result.get("unsupported_claims", [])
                }
            else:
                # Fallback parsing
                return {"score": 0.5, "reasoning": response.content, "unsupported_claims": []}

        except Exception as e:
            print(f"Error in faithfulness evaluation: {e}")
            return {"score": 0.5, "reasoning": f"Error: {str(e)}", "unsupported_claims": []}

    def evaluate_answer_relevance(
        self,
        query: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate if the answer is relevant to the query

        Answer relevance measures how well the generated answer addresses
        the specific question asked, regardless of factual correctness.

        Args:
            query: Original query
            answer: Generated answer

        Returns:
            Relevance score and details
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator assessing answer relevance.

Your task: Determine if the answer directly addresses the question asked.

Scoring Guide:
- 1.0: Answer directly and completely addresses the question
- 0.7-0.9: Answer addresses the question with minor irrelevant information
- 0.4-0.6: Answer partially addresses the question
- 0.1-0.3: Answer barely addresses the question
- 0.0: Answer is completely irrelevant to the question"""),
            ("user", """Question:
{query}

Answer to evaluate:
{answer}

Provide your evaluation in JSON format:
{{
    "score": <float between 0 and 1>,
    "reasoning": "<explanation>"
}}""")
        ])

        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "query": query,
                "answer": answer
            })

            result = extract_json_from_text(response.content)
            if result:
                return {
                    "score": float(result.get("score", 0.0)),
                    "reasoning": result.get("reasoning", "")
                }
            else:
                return {"score": 0.5, "reasoning": response.content}

        except Exception as e:
            print(f"Error in relevance evaluation: {e}")
            return {"score": 0.5, "reasoning": f"Error: {str(e)}"}

    def evaluate_context_utilization(
        self,
        answer: str,
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate how well the answer utilizes the retrieved contexts

        Context utilization measures whether the answer effectively uses
        the information available in the contexts.

        Args:
            answer: Generated answer
            contexts: Retrieved contexts

        Returns:
            Utilization score and details
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator assessing context utilization.

Your task: Determine how well the answer utilizes information from the provided contexts.

Scoring Guide:
- 1.0: Answer effectively synthesizes information from multiple contexts
- 0.7-0.9: Answer uses most relevant information from contexts
- 0.4-0.6: Answer uses some context information
- 0.1-0.3: Answer uses minimal context information
- 0.0: Answer doesn't use any context information"""),
            ("user", """Contexts:
{contexts}

Answer to evaluate:
{answer}

Provide your evaluation in JSON format:
{{
    "score": <float between 0 and 1>,
    "reasoning": "<explanation>",
    "contexts_used": [<indices of contexts used, e.g. [0, 1, 2]>]
}}""")
        ])

        try:
            contexts_text = "\n\n".join([f"Context {i}: {ctx}" for i, ctx in enumerate(contexts)])

            chain = prompt | self.llm
            response = chain.invoke({
                "contexts": contexts_text,
                "answer": answer
            })

            result = extract_json_from_text(response.content)
            if result:
                return {
                    "score": float(result.get("score", 0.0)),
                    "reasoning": result.get("reasoning", ""),
                    "contexts_used": result.get("contexts_used", [])
                }
            else:
                return {"score": 0.5, "reasoning": response.content, "contexts_used": []}

        except Exception as e:
            print(f"Error in context utilization evaluation: {e}")
            return {"score": 0.5, "reasoning": f"Error: {str(e)}", "contexts_used": []}

    def _calculate_traditional_metrics(
        self,
        generated: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Calculate traditional NLP metrics

        Args:
            generated: Generated answer
            reference: Ground truth answer

        Returns:
            Dictionary of metric scores
        """
        metrics = {}

        try:
            # ROUGE scores
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(reference, generated)

            metrics["rouge1_f1"] = rouge_scores['rouge1'].fmeasure
            metrics["rouge2_f1"] = rouge_scores['rouge2'].fmeasure
            metrics["rougeL_f1"] = rouge_scores['rougeL'].fmeasure

        except Exception as e:
            print(f"Error calculating ROUGE: {e}")

        try:
            # Semantic similarity using embeddings
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('all-MiniLM-L6-v2')

            emb1 = model.encode(generated, convert_to_tensor=True)
            emb2 = model.encode(reference, convert_to_tensor=True)

            similarity = util.cos_sim(emb1, emb2)
            metrics["semantic_similarity"] = float(similarity[0][0])

        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")

        return metrics

    def _calculate_overall_quality(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate overall quality score

        Args:
            metrics: Dictionary of individual metrics

        Returns:
            Overall quality score (0-1)
        """
        # Weighted combination of key metrics
        weights = {
            "faithfulness": 0.35,
            "answer_relevance": 0.35,
            "context_utilization": 0.30
        }

        score = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in metrics and isinstance(metrics[metric], dict):
                if "score" in metrics[metric]:
                    score += metrics[metric]["score"] * weight
                    total_weight += weight

        if total_weight > 0:
            return score / total_weight
        return 0.0

    def evaluate_batch(
        self,
        results: List[GenerationResult]
    ) -> Dict[str, Any]:
        """
        Evaluate multiple generation results

        Args:
            results: List of generation results

        Returns:
            Aggregated metrics
        """
        all_metrics = []
        detailed_results = []

        for result in results:
            metrics = self.evaluate_single(result)
            all_metrics.append(metrics)

            detailed_results.append({
                "query": result.query,
                "answer": result.generated_answer,
                "metrics": metrics
            })

        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_metrics)
        aggregated["detailed_results"] = detailed_results
        aggregated["num_queries"] = len(results)

        return aggregated

    def _aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics across multiple queries"""
        if not metrics_list:
            return {}

        aggregated = {}

        # Extract numeric scores
        numeric_metrics = {}
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, dict) and "score" in value:
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value["score"])
                elif isinstance(value, (int, float)):
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)

        # Calculate statistics
        for metric_name, values in numeric_metrics.items():
            aggregated[f"{metric_name}_mean"] = np.mean(values)
            aggregated[f"{metric_name}_std"] = np.std(values)
            aggregated[f"{metric_name}_median"] = np.median(values)

        return aggregated

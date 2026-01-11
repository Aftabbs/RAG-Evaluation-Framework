"""
Configuration management for RAG Evaluation Framework
"""
import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class EvaluationConfig(BaseModel):
    """Configuration for RAG evaluation"""

    # API Configuration
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = Field(default="mixtral-8x7b-32768")

    # Embedding Configuration
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # Evaluation Settings
    batch_size: int = Field(default=10)
    max_concurrent_requests: int = Field(default=5)

    # Retrieval Metrics
    k_values: list[int] = Field(default=[1, 3, 5, 10])

    # Generation Metrics Settings
    faithfulness_threshold: float = Field(default=0.7)
    relevance_threshold: float = Field(default=0.7)

    # Temperature for LLM evaluation
    evaluation_temperature: float = Field(default=0.0)

    class Config:
        validate_assignment = True


class RAGConfig(BaseModel):
    """Configuration for RAG system being evaluated"""

    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    top_k: int = Field(default=5)
    similarity_threshold: float = Field(default=0.7)



def get_config() -> EvaluationConfig:
    """Get evaluation configuration"""
    return EvaluationConfig()


def get_rag_config() -> RAGConfig:
    """Get RAG system configuration"""
    return RAGConfig()

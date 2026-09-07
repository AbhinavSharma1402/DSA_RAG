"""Configuration management for YouTube DSA RAG Tutor."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from .env file."""
    
    # LLM Configuration
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"  # Available Groq model
    llm_api_key: str = ""
    
    # Embedding Configuration
    # Preserve compatibility with the existing Qdrant vectors in the project.
    # The existing index was created with all-MiniLM-L6-v2 (384-dim).
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Local vector index Configuration
    vector_index_path: str = "vectors.npz"
    vector_index_name: str = "dsa_lectures_384"

    # Retrieval Configuration
    top_k: int = 6  # Initial retrieval count
    final_context_chunks: int = 5  # Chunks sent to LLM
    retrieval_threshold: float = 0.6  # MAX_DISTANCE: cosine distance cutoff
    
    # Query Rewriting Configuration
    enable_query_rewriting: bool = True
    rewriting_model: str = "openai/gpt-oss-120b"
    
    # Reranking configuration
    enable_reranker: bool = False  # Default to title-based reranking; cross-encoder optional
    reranker_model: Optional[str] = None

    # Title-based reranking (not cross-encoder)
    # Nudges chunks whose lecture title matches the query
    enable_title_boost: bool = True
    title_boost: float = 0.06  # Cosine distance penalty per matched word
    confident_distance: float = 0.45  # Threshold for high confidence
    
    # Query embedding prefix (for models that need it)
    # bge-m3 doesn't need one, but bge-*-en-v1.5 does
    embedding_query_prefix: str = ""
    
    # Conversation Configuration
    max_conversation_turns: int = 5
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Debug Configuration
    debug_rag: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Singleton instance
settings = Settings()

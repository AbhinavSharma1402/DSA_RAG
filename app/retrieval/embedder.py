"""Embedding service for converting text to vectors."""
from sentence_transformers import SentenceTransformer
from typing import List
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = None):
        """Initialize the embedding model."""
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text (document mode)."""
        if not text or not text.strip():
            return [0.0] * self.embedding_dim
        
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_query(self, query: str, prefix: str = "") -> List[float]:
        """Generate embedding for a query (with optional prefix)."""
        if not query or not query.strip():
            return [0.0] * self.embedding_dim
        
        # Some models (bge-*-en-v1.5) need an instruction prefix on queries only
        # bge-m3 doesn't need it, but including it doesn't hurt
        text = (prefix + query) if prefix else query
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.embedding_dim


# Singleton instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

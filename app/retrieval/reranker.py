"""Reranking module for improving retrieval result quality."""
from typing import List
from app.models.schemas import RetrievedChunk
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class Reranker:
    """Reranks retrieved chunks based on relevance to query."""
    
    def __init__(self, enable: bool = None):
        """Initialize reranker."""
        self.enable = enable if enable is not None else settings.enable_reranker
        
        if self.enable:
            try:
                from sentence_transformers.cross_encoders import CrossEncoder
                model_name = settings.reranker_model or "cross-encoder/ms-marco-MiniLM-L-12-v2"
                self.model = CrossEncoder(model_name)
                logger.info(f"Reranker initialized with model: {model_name}")
            except ImportError:
                logger.warning("sentence-transformers cross-encoder not available, disabling reranker")
                self.enable = False
                self.model = None
    
    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int = None) -> List[RetrievedChunk]:
        """
        Rerank chunks based on cross-encoder relevance score.
        
        If reranking is disabled, returns chunks sorted by original scores.
        """
        if not chunks:
            return chunks
        
        top_k = top_k or settings.final_context_chunks
        
        if not self.enable or not self.model:
            # Fall back to sorting by similarity score
            return sorted(chunks, key=lambda x: x.similarity_score, reverse=True)[:top_k]
        
        try:
            # Prepare pairs for cross-encoder
            pairs = [[query, chunk.text] for chunk in chunks]
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Update chunk scores and sort
            for chunk, score in zip(chunks, scores):
                chunk.similarity_score = float(score)
            
            reranked = sorted(chunks, key=lambda x: x.similarity_score, reverse=True)[:top_k]
            logger.debug(f"Reranked {len(chunks)} chunks, selected top {len(reranked)}")
            
            return reranked
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fall back to original scores
            return sorted(chunks, key=lambda x: x.similarity_score, reverse=True)[:top_k]


# Singleton instance
_reranker = None


def get_reranker() -> Reranker:
    """Get or create the reranker singleton."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker

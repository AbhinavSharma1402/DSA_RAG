"""Query rewriting for improving retrieval quality."""
from typing import Optional
from app.config import settings
from app.rag.prompt import get_query_rewriter_prompt
import logging

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Rewrites queries to improve retrieval quality."""
    
    def __init__(self, llm_provider=None):
        """Initialize query rewriter with LLM provider."""
        self.enabled = settings.enable_query_rewriting
        self.llm_provider = llm_provider
    
    def rewrite(self, query: str, conversation_context: str = "") -> Optional[str]:
        """
        Rewrite a query to improve retrieval.
        
        Returns:
            Rewritten query, or None if rewriting is disabled/failed
        """
        if not self.enabled:
            return None
        
        if not self.llm_provider:
            logger.warning("Query rewriter: LLM provider not initialized")
            return None
        
        try:
            prompt = get_query_rewriter_prompt(query, conversation_context)
            
            rewritten = self.llm_provider.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.3
            )
            
            rewritten = rewritten.strip()
            
            if rewritten == query:
                logger.debug("Query rewriter: no rewriting needed")
                return None
            
            logger.debug(f"Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten
            
        except Exception as e:
            logger.error(f"Error during query rewriting: {e}")
            return None
    
    def should_rewrite(self, query: str) -> bool:
        """Determine if query should be rewritten."""
        if not self.enabled:
            return False
        
        # Skip rewriting for very short or already clean queries
        if len(query) < 10:
            return False
        
        # Check for informal language patterns
        informal_patterns = ["bhai", "sir", "kaise", "kya", "aur", "me"]
        query_lower = query.lower()
        
        has_informal = any(pattern in query_lower for pattern in informal_patterns)
        
        return has_informal or "?" not in query

"""Title-based reranking (not cross-encoder) - matches actual ytscraper implementation."""
from typing import List, Set
from app.models.schemas import RetrievedChunk
from app.config import settings
import logging
import re

logger = logging.getLogger(__name__)

# Words that carry no topic signal (Hinglish scaffolding + boilerplate)
_STOP_WORDS = {
    "kaise", "kya", "hai", "hain", "me", "ka", "ki", "ke", "aur",
    "kab", "karte", "karna", "hota", "nikale", "solve", "kare",
    "chahiye", "use", "kahan", "se", "ko", "pehchane", "difference",
    "farak", "the", "a", "is", "in", "what", "how", "do", "to",
    "of", "for", "video", "dsa", "patterns", "pattern", "episode",
    "leetcode", "interview", "questions", "question", "master",
    "best", "explained",
}


def _stem(word: str) -> str:
    """Crude plural stripping - match 'hashmap' against 'HASHMAPS'."""
    for suffix in ("es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _extract_terms(text: str) -> Set[str]:
    """Extract meaningful terms from text (stop words removed, stemmed)."""
    return {
        _stem(w)
        for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in _STOP_WORDS and len(w) > 2
    }


def title_overlap(query: str, title: str) -> int:
    """How many meaningful query words appear in the lecture title."""
    return len(_extract_terms(query) & _extract_terms(title))


class TitleBoostReranker:
    """Reranks using title overlap (matches ytscraper approach)."""
    
    def __init__(self, enable: bool = None, title_boost: float = None):
        """Initialize reranker."""
        self.enable = enable if enable is not None else settings.enable_title_boost
        self.title_boost = title_boost if title_boost is not None else settings.title_boost
    
    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Rerank chunks using title overlap.
        
        Strategy: Nudge chunks whose lecture title mentions what was asked.
        Dense similarity over a 75-second ramble dilutes the topic badly.
        The title is the one place the topic is stated plainly.
        
        Adjustment: distance - TITLE_BOOST * overlap
        where overlap is the count of matching terms between query and title.
        """
        if not self.enable or not chunks:
            return chunks
        
        try:
            # Score each chunk: adjust distance by title overlap
            scored = []
            for chunk in chunks:
                overlap = title_overlap(query, chunk.video_title)
                adjusted_distance = chunk.similarity_score - (self.title_boost * overlap)
                scored.append((adjusted_distance, chunk))
            
            # Sort by adjusted distance (lower is better)
            scored.sort(key=lambda x: x[0])
            reranked = [chunk for _, chunk in scored]
            
            logger.debug(f"Title-boost reranking: {len(chunks)} chunks reranked using title overlap")
            return reranked
            
        except Exception as e:
            logger.error(f"Error during title-boost reranking: {e}")
            return chunks


# Singleton instance
_reranker = None


def get_title_boost_reranker() -> TitleBoostReranker:
    """Get or create the title-boost reranker singleton."""
    global _reranker
    if _reranker is None:
        _reranker = TitleBoostReranker()
    return _reranker

"""Confidence scoring based on retrieval quality."""
from typing import List
from app.models.schemas import RetrievedChunk
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Scores answer confidence based on retrieval quality."""
    
    def __init__(self, high_threshold: float = 0.45,
                 medium_threshold: float = 0.35,
                 low_threshold: float = 0.25):
        """Initialize confidence scorer with thresholds."""
        # Cosine similarity scoring: higher = better match
        # Refusal threshold: 0.25 (25% similarity = too weak to answer)
        # Low confidence: < 0.35 (35% similarity = weak but answerable)
        # Medium confidence: 0.35-0.45 (35-45% similarity = moderate)
        # High confidence: > 0.45 (45%+ similarity = strong match)
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold
    
    def score(self, chunks: List[RetrievedChunk], query: str = "") -> str:
        """Score confidence using cosine similarity semantics (higher = better)."""
        if not chunks:
            return "low"

        # Cosine similarity: range [0, 1], higher = better match
        scores = [chunk.similarity_score for chunk in chunks]
        best_similarity = max(scores)  # Higher is better
        avg_similarity = sum(scores) / len(scores)
        strong_results = sum(1 for s in scores if s >= 0.45)  # Strong matches

        if len(scores) > 1:
            variance = sum((s - avg_similarity) ** 2 for s in scores) / len(scores)
            consistency = 1 / (1 + variance)
        else:
            consistency = 1.0

        # Weighted confidence calculation favoring high similarity
        confidence_score = (
            0.45 * min(best_similarity, 1.0) +          # Best match weight
            0.25 * min(avg_similarity, 1.0) +            # Average similarity
            0.20 * min(strong_results / max(1, len(scores)), 1.0) +  # Strong result count
            0.10 * consistency                            # Consistency bonus
        )

        # Confidence levels based on similarity
        if best_similarity >= 0.60:
            confidence = "high"
        elif best_similarity >= 0.40:
            confidence = "medium"
        else:
            confidence = "low"

        logger.debug(
            f"Confidence scoring: best_sim={best_similarity:.3f}, avg_sim={avg_similarity:.3f}, "
            f"strong={strong_results}, consistency={consistency:.3f}, "
            f"final={confidence_score:.3f} -> {confidence}"
        )

        return confidence

    def should_refuse(self, chunks: List[RetrievedChunk]) -> bool:
        """Refuse when the strongest retrieval result is worse than the repo threshold."""
        if not chunks:
            logger.info("[SHOULD_REFUSE] No chunks provided, refusing")
            return True

        threshold = settings.retrieval_threshold  # Default 0.5 (but interpreted as minimum acceptable similarity now)
        top_similarity = chunks[0].similarity_score

        logger.info(f"[SHOULD_REFUSE] Top similarity score: {top_similarity:.4f}, Minimum threshold: {threshold}")
        logger.info(f"[SHOULD_REFUSE] Score interpretation: Higher is better (cosine similarity)")
        
        # FIXED LOGIC: Refuse if similarity is BELOW threshold (too weak)
        # Before: if top > threshold: refuse (WRONG - backwards logic)
        # After:  if top < threshold: refuse (CORRECT - refuse only if weak)
        should_refuse = top_similarity < threshold
        
        logger.info(f"[SHOULD_REFUSE] Check: {top_similarity:.4f} < {threshold} ? {should_refuse}")
        
        if should_refuse:
            logger.info(f"[SHOULD_REFUSE] YES - refusing because similarity {top_similarity:.4f} is below minimum {threshold}")
        else:
            logger.info(f"[SHOULD_REFUSE] NO - will answer because similarity {top_similarity:.4f} meets minimum {threshold}")
        
        return should_refuse
    
    def get_debug_info(self, chunks: List[RetrievedChunk]) -> dict:
        """Get debug information about confidence scoring."""
        if not chunks:
            return {"error": "no_chunks"}
        
        scores = [chunk.similarity_score for chunk in chunks]
        
        return {
            "top_score": scores[0],
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "num_chunks": len(chunks),
            "strong_results": sum(1 for s in scores if s > 0.5),
            "scores": scores[:5]  # Top 5 scores
        }

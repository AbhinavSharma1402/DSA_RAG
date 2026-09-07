"""Result fusion and deduplication for RAG retrieval."""
from typing import List, Dict
from app.models.schemas import RetrievedChunk
import logging

logger = logging.getLogger(__name__)


class ResultFusion:
    """Fuses results from multiple retrieval queries."""
    
    @staticmethod
    def merge_results(original_results: List[RetrievedChunk],
                     rewritten_results: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Merge results from original and rewritten queries.
        
        Strategy:
        1. Combine both result sets
        2. Deduplicate by chunk_id
        3. Preserve highest similarity score
        4. Re-rank by score
        5. Return merged results
        """
        if not rewritten_results:
            return original_results
        if not original_results:
            return rewritten_results
        
        # Create a dictionary to track chunks by ID
        merged: Dict[str, RetrievedChunk] = {}
        
        # Add original results
        for chunk in original_results:
            merged[chunk.chunk_id] = chunk
        
        # Merge rewritten results, keeping highest score
        for chunk in rewritten_results:
            if chunk.chunk_id in merged:
                # Keep the chunk with higher similarity score
                if chunk.similarity_score > merged[chunk.chunk_id].similarity_score:
                    merged[chunk.chunk_id] = chunk
            else:
                merged[chunk.chunk_id] = chunk
        
        # Sort by similarity score (descending)
        fused = sorted(merged.values(), key=lambda x: x.similarity_score, reverse=True)
        
        logger.debug(f"Fused {len(original_results)} + {len(rewritten_results)} "
                    f"results into {len(fused)} unique chunks")
        
        return fused
    
    @staticmethod
    def reciprocal_rank_fusion(original_results: List[RetrievedChunk],
                              rewritten_results: List[RetrievedChunk],
                              k: int = 60) -> List[RetrievedChunk]:
        """
        Apply Reciprocal Rank Fusion (RRF) to combine result sets.
        
        RRF score = sum(1 / (k + rank))
        where k is a constant (typically 60) and rank is position in result set.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}
        
        # Score original results
        for rank, chunk in enumerate(original_results, 1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1 / (k + rank)
            chunk_map[chunk.chunk_id] = chunk
        
        # Score rewritten results
        for rank, chunk in enumerate(rewritten_results, 1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0) + 1 / (k + rank)
            chunk_map[chunk.chunk_id] = chunk
        
        # Sort by RRF score
        sorted_chunks = sorted(
            chunk_map.items(),
            key=lambda x: rrf_scores[x[0]],
            reverse=True
        )
        
        # Update similarity scores with RRF scores for consistency
        result = []
        for chunk_id, chunk in sorted_chunks:
            chunk.similarity_score = rrf_scores[chunk_id]
            result.append(chunk)
        
        logger.debug(f"RRF fusion: {len(original_results)} + {len(rewritten_results)} "
                    f"-> {len(result)} chunks")
        
        return result


class Deduplicator:
    """Deduplicates chunks based on various strategies."""
    
    @staticmethod
    def deduplicate_by_id(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Deduplicate chunks by chunk_id, keeping first occurrence."""
        seen = set()
        result = []
        
        for chunk in chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                result.append(chunk)
        
        logger.debug(f"Deduplicated from {len(chunks)} to {len(result)} chunks")
        return result
    
    @staticmethod
    def deduplicate_by_timestamp(chunks: List[RetrievedChunk],
                                 tolerance_seconds: int = 30) -> List[RetrievedChunk]:
        """Deduplicate chunks from same video within tolerance window."""
        result = []
        seen_videos: Dict[str, List[int]] = {}
        
        for chunk in chunks:
            video_id = chunk.video_id
            
            if video_id not in seen_videos:
                seen_videos[video_id] = []
                result.append(chunk)
                seen_videos[video_id].append(chunk.timestamp)
            else:
                # Check if timestamp is too close to any existing timestamp
                is_duplicate = False
                for existing_ts in seen_videos[video_id]:
                    if abs(chunk.timestamp - existing_ts) <= tolerance_seconds:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    result.append(chunk)
                    seen_videos[video_id].append(chunk.timestamp)
        
        logger.debug(f"Timestamp dedup: {len(chunks)} -> {len(result)} chunks")
        return result
    
    @staticmethod
    def deduplicate_by_text_similarity(chunks: List[RetrievedChunk],
                                      similarity_threshold: float = 0.95) -> List[RetrievedChunk]:
        """Deduplicate chunks with very similar text content."""
        from difflib import SequenceMatcher
        
        result = []
        
        for chunk in chunks:
            is_duplicate = False
            
            for existing_chunk in result:
                # Quick check: same video and nearby timestamp
                if chunk.video_id == existing_chunk.video_id:
                    if abs(chunk.timestamp - existing_chunk.timestamp) < 60:
                        # Compute text similarity
                        matcher = SequenceMatcher(
                            None,
                            existing_chunk.text.lower(),
                            chunk.text.lower()
                        )
                        ratio = matcher.ratio()
                        
                        if ratio > similarity_threshold:
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                result.append(chunk)
        
        logger.debug(f"Text similarity dedup: {len(chunks)} -> {len(result)} chunks")
        return result

"""Context builder for constructing LLM input from retrieved chunks."""
from typing import List, Tuple
from app.models.schemas import RetrievedChunk, Citation
from app.rag.prompt import format_context_for_prompt
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds context from retrieved chunks for LLM input."""
    
    @staticmethod
    def build_context(chunks: List[RetrievedChunk], 
                      final_context_chunks: int = None) -> Tuple[str, List[Citation]]:
        """
        Build context string and citations from retrieved chunks.
        
        Returns:
            Tuple of (formatted_context, citation_list)
        """
        final_context_chunks = final_context_chunks or settings.final_context_chunks
        
        if not chunks:
            return "", []
        
        # Select final chunks
        selected_chunks = chunks[:final_context_chunks]
        
        # Generate citations
        citations = []
        for chunk in selected_chunks:
            citation = Citation(
                chunk_id=chunk.chunk_id,
                video_id=chunk.video_id,
                video_title=chunk.video_title,
                timestamp=chunk.timestamp,
                timestamp_display=chunk.timestamp_display(),
                url=ContextBuilder.generate_youtube_url(
                    chunk.video_id,
                    chunk.timestamp
                )
            )
            citations.append(citation)
        
        # Format context
        context = format_context_for_prompt(selected_chunks)
        
        logger.debug(f"Built context with {len(selected_chunks)} chunks and {len(citations)} citations")
        
        return context, citations
    
    @staticmethod
    def generate_youtube_url(video_id: str, timestamp: int, rewind_seconds: int = 5) -> str:
        """
        Generate a timestamp-aware YouTube URL.
        
        Args:
            video_id: YouTube video ID
            timestamp: Timestamp in seconds
            rewind_seconds: Rewind from timestamp (default 5 for better context)
        
        Returns:
            YouTube URL with timestamp parameter
        """
        # Rewind for better context
        adjusted_timestamp = max(0, timestamp - rewind_seconds)
        return f"https://www.youtube.com/watch?v={video_id}&t={adjusted_timestamp}s"
    
    @staticmethod
    def extract_source_ids_from_answer(answer: str) -> List[str]:
        """
        Extract source IDs from LLM answer.
        
        Expected format: "SOURCES: source_1, source_2"
        """
        if "SOURCES:" not in answer:
            return []
        
        try:
            sources_part = answer.split("SOURCES:")[-1].strip()
            
            if not sources_part or sources_part.lower() == "":
                return []
            
            # Parse source IDs
            source_ids = [s.strip() for s in sources_part.split(",")]
            source_ids = [s for s in source_ids if s]  # Remove empty strings
            
            return source_ids
        except Exception as e:
            logger.warning(f"Error parsing source IDs: {e}")
            return []
    
    @staticmethod
    def clean_answer(answer: str) -> str:
        """Remove source metadata from answer text."""
        if "SOURCES:" in answer:
            return answer.split("SOURCES:")[0].strip()
        return answer.strip()

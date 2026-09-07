"""Pydantic models and schemas for the RAG system."""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class RetrievedChunk(BaseModel):
    """A single chunk retrieved from Qdrant."""
    chunk_id: str
    text: str
    video_id: str
    video_title: str
    topic: Optional[str] = None
    timestamp: int  # seconds
    duration: Optional[int] = None
    source_url: str
    similarity_score: float = 0.0
    
    def timestamp_display(self) -> str:
        """Convert seconds to HH:MM:SS or MM:SS format."""
        hours = self.timestamp // 3600
        minutes = (self.timestamp % 3600) // 60
        seconds = self.timestamp % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"


class Citation(BaseModel):
    """A citation for an answer with timestamp and video link."""
    chunk_id: str
    video_id: str
    video_title: str
    timestamp: int
    timestamp_display: str
    url: str
    

class ChatRequest(BaseModel):
    """Request body for /api/chat endpoint."""
    query: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    topic: Optional[str] = "All"
    mode: str = Field(default="lecture", pattern="^(lecture|tutor)$")


class ChatResponse(BaseModel):
    """Response body for /api/chat endpoint."""
    answer: str
    confidence: str = Field(pattern="^(high|medium|low)$")
    mode: str
    rewritten_query: Optional[str] = None
    sources: List[Citation] = []
    retrieved_chunks: int = 0
    debug_info: Optional[dict] = None


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""
    timestamp: datetime
    role: str = Field(pattern="^(user|assistant)$")
    content: str
    query: Optional[str] = None
    answer: Optional[str] = None
    sources: Optional[List[Citation]] = None


class Conversation(BaseModel):
    """A conversation history."""
    conversation_id: str
    created_at: datetime
    turns: List[ConversationTurn] = []
    
    def add_turn(self, role: str, content: str, query: Optional[str] = None,
                 answer: Optional[str] = None, sources: Optional[List[Citation]] = None) -> None:
        """Add a turn to the conversation."""
        turn = ConversationTurn(
            timestamp=datetime.utcnow(),
            role=role,
            content=content,
            query=query,
            answer=answer,
            sources=sources
        )
        self.turns.append(turn)
    
    def get_recent_context(self, max_turns: int = 5) -> str:
        """Get recent conversation context as a string."""
        recent = self.turns[-max_turns:]
        context = []
        for turn in recent:
            context.append(f"{turn.role.upper()}: {turn.content}")
        return "\n".join(context)

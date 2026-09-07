"""Conversation memory management."""
from typing import Dict, Optional
from datetime import datetime
from app.models.schemas import Conversation, Citation
from app.config import settings
import logging
import uuid

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manages conversation history in memory."""
    
    def __init__(self):
        """Initialize conversation store."""
        self.conversations: Dict[str, Conversation] = {}
    
    def create_conversation(self) -> str:
        """Create a new conversation and return its ID."""
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = Conversation(
            conversation_id=conversation_id,
            created_at=datetime.utcnow()
        )
        logger.debug(f"Created new conversation: {conversation_id}")
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)
    
    def add_user_message(self, conversation_id: str, query: str) -> bool:
        """Add a user message to conversation."""
        conv = self.get_conversation(conversation_id)
        if not conv:
            logger.warning(f"Conversation {conversation_id} not found")
            return False
        
        conv.add_turn(role="user", content=query, query=query)
        logger.debug(f"Added user message to conversation {conversation_id}")
        return True
    
    def add_assistant_message(self, conversation_id: str, answer: str,
                             query: Optional[str] = None,
                             sources: Optional[list] = None) -> bool:
        """Add an assistant message to conversation."""
        conv = self.get_conversation(conversation_id)
        if not conv:
            logger.warning(f"Conversation {conversation_id} not found")
            return False
        
        conv.add_turn(
            role="assistant",
            content=answer,
            query=query,
            answer=answer,
            sources=sources or []
        )
        logger.debug(f"Added assistant message to conversation {conversation_id}")
        return True
    
    def get_context(self, conversation_id: str, max_turns: int = None) -> str:
        """Get recent conversation context as string."""
        max_turns = max_turns or settings.max_conversation_turns
        
        conv = self.get_conversation(conversation_id)
        if not conv:
            return ""
        
        return conv.get_recent_context(max_turns)
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.debug(f"Cleared conversation {conversation_id}")
            return True
        return False
    
    def get_all_conversations(self) -> Dict[str, Conversation]:
        """Get all conversations (for debugging/monitoring)."""
        return self.conversations.copy()


# Singleton instance
_conversation_memory = None


def get_conversation_memory() -> ConversationMemory:
    """Get or create the conversation memory singleton."""
    global _conversation_memory
    if _conversation_memory is None:
        _conversation_memory = ConversationMemory()
    return _conversation_memory

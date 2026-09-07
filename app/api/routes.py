"""FastAPI routes for the RAG system."""
from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import ChatRequest, ChatResponse
from app.rag.pipeline import get_rag_pipeline
from app.retrieval.qdrant import get_vector_store
from app.memory.conversation import get_conversation_memory
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a user chat message and return an answer with citations.
    
    Request:
        - query: The user's question
        - conversation_id: Optional conversation ID (creates new if not provided)
        - topic: Optional topic filter (default: "All")
        - mode: "lecture" or "tutor" (default: "lecture")
    
    Response:
        - answer: The generated answer
        - confidence: "high", "medium", or "low"
        - rewritten_query: The rewritten query if applicable
        - sources: List of citations with timestamps and YouTube URLs
        - retrieved_chunks: Number of chunks used
    """
    try:
        # Validate input
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if len(request.query) > 2000:
            raise HTTPException(status_code=400, detail="Query too long (max 2000 chars)")
        
        if request.mode not in ["lecture", "tutor"]:
            raise HTTPException(status_code=400, detail="Mode must be 'lecture' or 'tutor'")
        
        # Get RAG pipeline and process
        pipeline = get_rag_pipeline()
        
        response = pipeline.process_query(
            query=request.query,
            conversation_id=request.conversation_id,
            topic=request.topic if request.topic != "All" else None,
            mode=request.mode,
            debug=False
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/topics")
async def get_topics():
    """Get available topics for filtering."""
    try:
        vector_store = get_vector_store()
        topics = vector_store.get_available_topics()
        
        return {
            "topics": topics,
            "count": len(topics)
        }
    except Exception as e:
        logger.error(f"Error fetching topics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch topics")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        vector_store = get_vector_store()
        is_healthy = vector_store.health_check()
        
        if not is_healthy:
            raise HTTPException(status_code=503, detail="Local vector index unavailable")
        
        return {
            "status": "healthy",
            "service": "youtube-dsa-rag-tutor",
            "vector_store": "vectors.npz"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.post("/conversations")
async def create_conversation():
    """Create a new conversation."""
    try:
        memory = get_conversation_memory()
        conversation_id = memory.create_conversation()
        
        return {
            "conversation_id": conversation_id,
            "message": "Conversation created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to create conversation")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    try:
        memory = get_conversation_memory()
        success = memory.clear_conversation(conversation_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "message": "Conversation deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation")


@router.get("/debug/info")
async def debug_info():
    """Get debug information about the system."""
    try:
        vector_store = get_vector_store()
        collection_info = vector_store.get_collection_info()
        
        return {
            "collection": collection_info,
            "configuration": {
                "top_k": 8,  # settings.top_k
                "final_context_chunks": 5,  # settings.final_context_chunks
                "retrieval_threshold": 0.5,  # settings.retrieval_threshold
                "enable_query_rewriting": True,  # settings.enable_query_rewriting
                "enable_reranker": False,  # settings.enable_reranker
            }
        }
    except Exception as e:
        logger.error(f"Error getting debug info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get debug info")

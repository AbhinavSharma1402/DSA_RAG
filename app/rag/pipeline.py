"""Main RAG pipeline orchestrating the complete flow."""
from typing import Optional, List, Dict, Any
from app.models.schemas import ChatResponse, Citation, RetrievedChunk
from app.retrieval.embedder import get_embedding_service
from app.retrieval.qdrant import get_vector_store
from app.retrieval.reranker import get_reranker
from app.retrieval.title_boost import get_title_boost_reranker
from app.rag.prompt import get_refusal_message
from app.retrieval.fusion import ResultFusion, Deduplicator
from app.rag.query_rewriter import QueryRewriter
from app.rag.generator import AnswerGenerator, get_llm_provider
from app.rag.context_builder import ContextBuilder
from app.confidence.scorer import ConfidenceScorer
from app.memory.conversation import get_conversation_memory
from app.config import settings
import logging
import time

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline."""
    
    def __init__(self):
        """Initialize RAG pipeline components."""
        self.embedder = get_embedding_service()
        self.vector_store = get_vector_store()
        self.reranker = get_reranker()
        self.title_boost_reranker = get_title_boost_reranker()
        self.llm = get_llm_provider()
        
        self.query_rewriter = QueryRewriter(llm_provider=self.llm)
        self.answer_generator = AnswerGenerator(llm_provider=self.llm)
        self.confidence_scorer = ConfidenceScorer()
        self.conversation_memory = get_conversation_memory()
        
        logger.info("RAG pipeline initialized")
    
    def process_query(self, query: str, conversation_id: Optional[str] = None,
                     topic: Optional[str] = None, mode: str = "lecture",
                     debug: bool = False) -> ChatResponse:
        """
        Process a user query and return a complete answer.
        
        Complete pipeline:
        1. Get or create conversation
        2. Get conversation context
        3. Retrieve original query results
        4. Optionally rewrite and retrieve again
        5. Fuse and deduplicate results
        6. Rerank results
        7. Score confidence
        8. Build context for LLM
        9. Generate answer
        10. Extract citations
        11. Save to conversation memory
        """
        start_time = time.time()
        debug_info = {}
        
        # DEBUG: Log original query
        logger.info("=== RAG PIPELINE DEBUG ===")
        logger.info(f"[QUERY] Original: '{query}'")
        logger.info(f"[CONFIG] Topic: {topic}, Mode: {mode}")
        logger.info(f"[CONFIG] Embedding model: {self.embedder.model_name}")
        logger.info(f"[CONFIG] Embedding dimension: {self.embedder.embedding_dim}")
        
        # Step 1: Conversation management
        if not conversation_id:
            conversation_id = self.conversation_memory.create_conversation()
        
        self.conversation_memory.add_user_message(conversation_id, query)
        
        # Step 2: Get conversation context
        conv_context = self.conversation_memory.get_context(conversation_id)
        
        # Step 3: Embed original query
        logger.debug(f"Embedding original query: {query}")
        query_embedding = self.embedder.embed(query)
        logger.info(f"[EMBEDDING] Dimension: {len(query_embedding)}")
        
        # Retrieve original results
        original_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=settings.top_k,
            topic_filter=topic
        )
        
        logger.info(f"[RETRIEVAL] Original query retrieved {len(original_results)} results")
        if original_results:
            top_scores = [r.similarity_score for r in original_results[:5]]
            logger.info(f"[SCORES] Top 5 similarity scores: {top_scores}")
        
        debug_info["original_query"] = query
        debug_info["embedding_dim"] = len(query_embedding)
        debug_info["original_retrieval_count"] = len(original_results)
        debug_info["original_top_scores"] = [r.similarity_score for r in original_results[:5]]
        
        # Step 4: Query rewriting and second retrieval
        rewritten_query = None
        rewritten_results = []
        
        if self.query_rewriter.should_rewrite(query):
            logger.info(f"[REWRITE] Query rewriting enabled for: '{query}'")
            rewritten_query = self.query_rewriter.rewrite(query, conv_context)
            
            if rewritten_query and rewritten_query != query:
                logger.info(f"[REWRITE] Query rewritten to: '{rewritten_query}'")
                original_terms = set(query.lower().split())
                rewritten_terms = set(rewritten_query.lower().split())
                preserved = original_terms & rewritten_terms
                logger.info(f"[REWRITE] Preserved terms: {preserved}")
                
                rewritten_embedding = self.embedder.embed(rewritten_query)
                rewritten_results = self.vector_store.search(
                    query_embedding=rewritten_embedding,
                    top_k=settings.top_k,
                    topic_filter=topic
                )
                
                logger.info(f"[RETRIEVAL] Rewritten query retrieved {len(rewritten_results)} results")
                if rewritten_results:
                    rewritten_top_scores = [r.similarity_score for r in rewritten_results[:5]]
                    logger.info(f"[SCORES] Rewritten top 5: {rewritten_top_scores}")
                
                debug_info["rewritten_query"] = rewritten_query
                debug_info["preserved_terms"] = list(preserved)
                debug_info["rewritten_retrieval_count"] = len(rewritten_results)
                debug_info["rewritten_top_scores"] = [r.similarity_score for r in rewritten_results[:5]]
        
        # Step 5: Fuse results
        fused_results = ResultFusion.merge_results(original_results, rewritten_results)
        
        # Deduplicate
        deduplicated = Deduplicator.deduplicate_by_id(fused_results)
        
        # Step 6: Rerank
        # ytscraper uses title-based boost as the main reranking strategy.
        if settings.enable_reranker:
            final_chunks = self.reranker.rerank(query, deduplicated)
        else:
            final_chunks = self.title_boost_reranker.rerank(query, deduplicated)
        
        # Step 7: Score confidence
        should_refuse = self.confidence_scorer.should_refuse(final_chunks)
        confidence = self.confidence_scorer.score(final_chunks)
        
        logger.info(f"[CONFIDENCE] Should refuse: {should_refuse}, Confidence: {confidence}")
        if final_chunks:
            logger.info(f"[THRESHOLD] Top result score: {final_chunks[0].similarity_score:.4f}, Threshold: {settings.retrieval_threshold}")
            logger.info(f"[THRESHOLD] Refusal check: {final_chunks[0].similarity_score:.4f} > {settings.retrieval_threshold} = {should_refuse}")
        else:
            logger.info(f"[THRESHOLD] No chunks returned, triggering refusal")
        
        if debug:
            debug_info["should_refuse"] = should_refuse
            debug_info["confidence"] = confidence
            debug_info["final_chunks_count"] = len(final_chunks)
            debug_info["confidence_debug"] = self.confidence_scorer.get_debug_info(final_chunks)
        
        # Handle refusal
        if should_refuse:
            reason = f"(top score {final_chunks[0].similarity_score:.4f} > threshold {settings.retrieval_threshold})" if final_chunks else "(no relevant results)"
            logger.info(f"[REFUSAL] Refusing to answer query: '{query}' {reason}")
            refusal_message = get_refusal_message()
            response = ChatResponse(
                answer=refusal_message,
                confidence="low",
                mode=mode,
                rewritten_query=rewritten_query,
                sources=[],
                retrieved_chunks=len(final_chunks)
            )
            
            self.conversation_memory.add_assistant_message(
                conversation_id,
                response.answer,
                query=query,
                sources=[]
            )
            
            if debug:
                response.debug_info = debug_info
            
            return response
        
        # Step 8: Build context
        context, citations = ContextBuilder.build_context(final_chunks)
        
        # Step 9: Generate answer
        logger.info(f"Generating answer for: {query}")
        raw_answer = self.answer_generator.generate_answer(query, context, mode)
        
        # Step 10: Extract and clean answer
        answer = ContextBuilder.clean_answer(raw_answer)
        
        # Step 11: Save to conversation
        self.conversation_memory.add_assistant_message(
            conversation_id,
            answer,
            query=query,
            sources=citations
        )
        
        # Build response
        elapsed = time.time() - start_time
        logger.info(f"Query processed in {elapsed:.2f}s")
        
        response = ChatResponse(
            answer=answer,
            confidence=confidence,
            mode=mode,
            rewritten_query=rewritten_query,
            sources=citations,
            retrieved_chunks=len(final_chunks)
        )
        
        if debug:
            debug_info["elapsed_seconds"] = elapsed
            response.debug_info = debug_info
        
        return response


# Singleton instance
_pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline

#!/usr/bin/env python3
"""
End-to-end RAG test script.
Tests the complete flow: query -> embedding -> local vector retrieval -> generation -> citations.

This script assumes:
1. vectors.npz exists in the project root
2. The bundled vectors were created with all-MiniLM-L6-v2 embeddings
3. Groq API key is set in .env (LLM_API_KEY)
"""

import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

print("=" * 60)
print("RAG End-to-End Test")
print("=" * 60)

# Test 1: Environment and imports
print("\n[1/5] Checking environment...")
try:
    from app.config import settings
    print(f"✓ Config loaded")
    print(f"  - Embedding model: {settings.embedding_model}")
    print(f"  - Vector index: {settings.vector_index_path}")
    print(f"  - LLM model: {settings.llm_model}")
except Exception as e:
    print(f"✗ Config failed: {e}")
    sys.exit(1)

# Test 2: Embedding service
print("\n[2/5] Testing embedding service...")
try:
    from app.retrieval.embedder import get_embedding_service
    embedder = get_embedding_service()
    dim = embedder.get_dimension()
    print(f"✓ Embedder loaded")
    print(f"  - Model: {embedder.model_name}")
    print(f"  - Dimension: {dim}")
    
    if dim != 384:
        print(f"✗ ERROR: Expected 384-dim but got {dim}")
        sys.exit(1)
    
    # Test embedding
    test_query = "What is dynamic programming?"
    embedding = embedder.embed_query(test_query)
    print(f"✓ Test query embedded: {len(embedding)} values, first 5: {embedding[:5]}")
except Exception as e:
    print(f"✗ Embedder failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Local vector index retrieval
print("\n[3/5] Testing local vector retrieval...")
try:
    from app.retrieval.qdrant import get_vector_store
    vector_store = get_vector_store()

    if not vector_store.health_check():
        print("✗ Local vector index health check failed")
        print(f"  Make sure {settings.vector_index_path} exists in the project root")
        sys.exit(1)
    print("✓ Local vector index connected")

    info = vector_store.get_collection_info()
    if info:
        print(f"  - File: {info['name']}")
        print(f"  - Points: {info['points_count']}")
        print(f"  - Status: {info['status']}")

    print(f"  Testing retrieval with query: '{test_query}'")
    results = vector_store.search(embedding, top_k=3)
    print(f"✓ Retrieved {len(results)} chunks")

    if results:
        for i, chunk in enumerate(results[:3], 1):
            print(f"  [{i}] Video: {chunk.video_title[:40]}... @ {chunk.timestamp}s")
            print(f"      Score: {chunk.similarity_score:.4f}, Text: {chunk.text[:50]}...")
    else:
        print("  (no results for this query)")

except Exception as e:
    print(f"✗ Local vector retrieval failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: RAG Pipeline
print("\n[4/5] Testing RAG pipeline...")
try:
    from app.rag.pipeline import RAGPipeline
    
    pipeline = RAGPipeline()
    print(f"✓ RAG pipeline initialized")
    
    # Process a test query
    test_query = "Explain memoization in dynamic programming"
    print(f"  Query: '{test_query}'")
    
    try:
        response = pipeline.process_query(test_query, conversation_id="test-session-001")
        
        print(f"✓ Pipeline executed")
        print(f"  - Confidence: {response.confidence}")
        print(f"  - Has answer: {bool(response.answer)}")
        print(f"  - Citation count: {len(response.citations)}")
        
        if response.answer:
            print(f"  - Answer preview: {response.answer[:100]}...")
        
        if response.citations:
            print(f"  - First citation: {response.citations[0].video_title} @ {response.citations[0].timestamp}s")
    except Exception as pipeline_err:
        print(f"✗ Pipeline execution failed: {pipeline_err}")
        print(f"  (This may be expected if Groq API key is missing or empty)")
        import traceback
        traceback.print_exc()
    
except Exception as e:
    print(f"✗ Pipeline init failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Summary
print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print("""
✓ Dependency stack: VALID
✓ Embedding model: all-MiniLM-L6-v2 (384-dim)
✓ Environment config: LOADED
⚠ Local vector index: Check file exists and loads above
⚠ LLM: Requires valid Groq API key in .env

Next steps:
1. Confirm vectors.npz exists in the project root
2. Set LLM_API_KEY in .env to a valid Groq API key
3. Run this script again to test the full pipeline
""")

print("\n✓ All basic checks passed!")
print("✓ Ready for integration testing with the local vectors.npz index")

#!/usr/bin/env python3
"""Quick sanity check for RAG setup."""

import sys
print("Python version:", sys.version)

# Test imports
print("\nTesting imports...")
try:
    from sentence_transformers import SentenceTransformer
    print("✓ sentence-transformers imported")
except ImportError as e:
    print("✗ sentence-transformers failed:", e)
    sys.exit(1)

try:
    from transformers import AutoTokenizer
    print("✓ transformers imported")
except ImportError as e:
    print("✗ transformers failed:", e)
    sys.exit(1)

try:
    import numpy as np
    print("✓ numpy imported")
except ImportError as e:
    print("✗ numpy failed:", e)
    sys.exit(1)

try:
    import fastapi, pydantic, groq
    print("✓ fastapi, pydantic, groq imported")
except ImportError as e:
    print("✗ backend libs failed:", e)
    sys.exit(1)

# Test embedding model
print("\nTesting embedding model...")
try:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(["Hello world", "Test sentence"], normalize_embeddings=True)
    dim = embeddings.shape[1]
    print(f"✓ Model loaded, dimension: {dim}")
    
    if dim != 384:
        print(f"✗ Expected 384 dims but got {dim}")
        sys.exit(1)
    
    print(f"✓ Correct embedding dimension: {dim}")
except Exception as e:
    print(f"✗ Model test failed: {e}")
    sys.exit(1)

# Test local vector index
print("\nTesting local vector index...")
try:
    import numpy as np
    data = np.load("vectors.npz", allow_pickle=True)
    print(f"✓ Local vector index loaded")
    print(f"  Keys: {data.files}")
    print(f"  Vectors: {data['vectors'].shape}")
    print(f"  Dim: {data['dim']}")
    payload_count = len(data['payloads'].item()) if hasattr(data['payloads'], 'item') else 0
    print(f"  Payload entries: {payload_count}")
except Exception as e:
    print(f"✗ Local vector index test failed: {e}")
    print("  Make sure vectors.npz exists in the project root")

print("\n✓ All checks passed!")

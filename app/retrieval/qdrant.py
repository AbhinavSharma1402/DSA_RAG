"""Local vector-index retrieval backed by the bundled vectors.npz archive."""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import settings
from app.models.schemas import RetrievedChunk
from app.retrieval.embedder import get_embedding_service
import logging

logger = logging.getLogger(__name__)


class LocalVectorStore:
    """Search the bundled vectors.npz archive without needing Qdrant."""

    def __init__(self, index_path: str = None):
        self.embedding_service = get_embedding_service()
        self.index_path = index_path or settings.vector_index_path
        self.index_file = self._resolve_index_path(self.index_path)
        self.vectors: Optional[np.ndarray] = None
        self.payloads: List[Dict[str, Any]] = []
        self.dim: int = self.embedding_service.embedding_dim
        self._load_index()

    def _resolve_index_path(self, index_path: str) -> str:
        path = Path(index_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return str(path)

    def _load_index(self) -> None:
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"Vector index not found: {self.index_file}")

        data = np.load(self.index_file, allow_pickle=True)
        self.vectors = np.asarray(data["vectors"], dtype=np.float32)
        self.dim = int(data["dim"]) if "dim" in data.files else self.vectors.shape[1]

        raw_payloads = data["payloads"]
        payload_string = raw_payloads.item() if hasattr(raw_payloads, "item") else str(raw_payloads)
        parsed = json.loads(payload_string)
        if isinstance(parsed, dict):
            self.payloads = [parsed]
        else:
            self.payloads = parsed

        logger.info("Loaded local vector index: %s (%d vectors, %d-dim)", self.index_file, len(self.payloads), self.dim)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector.astype(np.float32)
        return (vector / norm).astype(np.float32)

    def search(self, query_embedding: List[float], top_k: int = None, topic_filter: Optional[str] = None) -> List[RetrievedChunk]:
        top_k = top_k or settings.top_k
        if self.vectors is None or len(self.vectors) == 0:
            logger.warning("No vectors loaded from local index")
            return []

        try:
            query_vector = np.asarray(query_embedding, dtype=np.float32)
            if query_vector.shape[0] != self.dim:
                raise ValueError(f"Query dimension mismatch: expected {self.dim}, got {query_vector.shape[0]}")

            normalized_query = self._normalize(query_vector)
            normalized_vectors = np.vstack([self._normalize(v) for v in self.vectors])

            scores = normalized_vectors @ normalized_query

            candidate_indices = list(range(len(self.payloads)))
            if topic_filter and topic_filter != "All":
                candidate_indices = [
                    idx for idx in candidate_indices
                    if str(self.payloads[idx].get("topic", "")).lower() == str(topic_filter).lower()
                ]

            if not candidate_indices:
                logger.info("No local vector entries match topic filter: %s", topic_filter)
                return []

            ranked = sorted(candidate_indices, key=lambda idx: float(scores[idx]), reverse=True)[:top_k]
            chunks: List[RetrievedChunk] = []
            
            # DEBUG: Log all raw scores before filtering
            logger.info(f"[RAW SCORES] All {len(ranked)} top-k results (no threshold filtering yet):")
            for rank, idx in enumerate(ranked[:5], 1):
                raw_score = float(scores[idx])
                video_id = self.payloads[idx].get("video_id", "UNKNOWN")
                title = self.payloads[idx].get("video_title", "UNKNOWN")[:40]
                logger.info(f"  [{rank}] score={raw_score:.4f}, video_id={video_id}, title={title}")
            
            for idx in ranked:
                payload = self.payloads[idx]
                score = float(scores[idx])
                start_sec = int(payload.get("start_sec", payload.get("timestamp", 0)))
                end_sec = payload.get("end_sec")
                duration_value = payload.get("duration")
                if duration_value is None and end_sec is not None:
                    duration_value = int(end_sec) - int(start_sec)

                source_url = payload.get(
                    "source_url",
                    f"https://www.youtube.com/watch?v={payload.get('video_id', '')}&t={int(start_sec)}s"
                )

                chunk = RetrievedChunk(
                    chunk_id=payload.get("chunk_id", str(idx)),
                    text=payload.get("text", ""),
                    video_id=payload.get("video_id", ""),
                    video_title=payload.get("video_title", "Unknown"),
                    topic=payload.get("topic"),
                    timestamp=start_sec,
                    duration=duration_value,
                    source_url=source_url,
                    similarity_score=score
                )
                
                # DEBUG: Validate payload contains expected fields
                if not chunk.text:
                    logger.warning(f"[PAYLOAD] Chunk {chunk.chunk_id}: empty text")
                if not chunk.video_id:
                    logger.warning(f"[PAYLOAD] Chunk {chunk.chunk_id}: missing video_id")
                if chunk.timestamp is None or chunk.timestamp < 0:
                    logger.warning(f"[PAYLOAD] Chunk {chunk.chunk_id}: invalid timestamp")
                
                chunks.append(chunk)

            logger.debug("Retrieved %d local chunks for query", len(chunks))
            if chunks:
                logger.info(f"[PAYLOAD SAMPLE] Top chunk: video_id={chunks[0].video_id}, timestamp={chunks[0].timestamp}, text_len={len(chunks[0].text)}")
            return chunks

        except Exception as e:
            logger.error("Error searching local vector index: %s", e, exc_info=True)
            return []

    def health_check(self) -> bool:
        return os.path.exists(self.index_file) and self.vectors is not None and len(self.vectors) > 0

    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        return {
            "name": os.path.basename(self.index_file),
            "points_count": len(self.payloads),
            "vectors_count": len(self.payloads),
            "status": "local-npz"
        }

    def get_available_topics(self) -> List[str]:
        topics = sorted({str(item.get("topic")) for item in self.payloads if item.get("topic")})
        return ["All", *topics]


_vector_store = None


def get_vector_store() -> LocalVectorStore:
    """Get or create the local vectors.npz-backed retrieval store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = LocalVectorStore()
    return _vector_store

import logging
from typing import Dict, Any, List, Optional

from utils.vector_store import search_similar_chunks

logger = logging.getLogger(__name__)

class RetrievalAgent:
    """
    Responsible for retrieving the most relevant context chunks for a given query
    filtered to a specific document (doc_id).
    """

    def __init__(
        self,
        default_k: int = 5,
        min_score_threshold: float = 0.18,          # depends on your embedding model & normalization
        score_field: str = "distance",              # or "similarity" — adjust according to vector_store
    ):
        """
        Args:
            default_k:               default number of chunks to retrieve
            min_score_threshold:     minimum similarity/distance score to accept
            score_field:             name of the score field returned by search_similar_chunks
        """
        self.default_k = default_k
        self.min_score_threshold = min_score_threshold
        self.score_field = score_field

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected message format (from Streamlit / user):
        {
            "payload": {
                "query": str,
                "doc_id": str,
                # optional:
                "k": int | None,
                "min_score": float | None,
                "filter": dict | None,      # e.g. {"file_name": "...", "chunk_index": ...}
            },
            ...
        }

        Returns something like:
        {
            "status":        "success" | "error",
            "query":         str,
            "chunks":        List[Dict],           # best format for downstream LLM
            "raw_results":   List[Dict] | List,    # original return value (for debugging)
            "retrieved_count": int,
            "message":       str | None,
            "error":         str | None
        }
        """
        try:
            payload = message.get("payload", {})
            query = payload.get("query", "").strip()
            doc_id = payload.get("doc_id")

            if not query:
                return self._error_response("No query provided")

            if not doc_id:
                return self._error_response("No doc_id provided — retrieval must be document-specific")

            # Allow overriding defaults per request
            k = payload.get("k", self.default_k)
            min_score = payload.get("min_score", self.min_score_threshold)
            metadata_filter = payload.get("filter", None)

            logger.info(f"Retrieval started | doc_id={doc_id[:8]}... | query='{query[:60]}...' | k={k}")

            # ── Perform vector search ────────────────────────────────────────
            search_result = search_similar_chunks(
                doc_id=doc_id,
                query=query,
                k=k,
                # pass filter if your vector store supports metadata filtering
                # filter=metadata_filter,
            )

            # ── Normalize result format ──────────────────────────────────────
            # We want downstream (LLM) to receive clean list of dicts
            formatted_chunks = []

            # Handle different possible return shapes from search_similar_chunks
            if isinstance(search_result, list):
                for item in search_result:

                    # Case 1: returns plain strings
                    if isinstance(item, str):
                        formatted_chunks.append({
                            "text": item,
                            "score": None,
                            "metadata": {}
                        })

                    # Case 2: returns tuples (text, score)
                    elif isinstance(item, tuple) and len(item) >= 2:
                        text, score = item[:2]
                        meta = item[2] if len(item) > 2 else {}
                        formatted_chunks.append({
                            "text": text,
                            "score": score,
                            "metadata": meta or {}
                        })

                    # Case 3: returns dicts (most common with Chroma, Qdrant, etc.)
                    elif isinstance(item, dict):
                        text = (
                            item.get("text")
                            or item.get("content")
                            or item.get("page_content")
                            or item.get("document", "")
                        )
                        score = item.get(self.score_field) or item.get("score") or item.get("distance")
                        meta = item.get("metadata", {}) or {}
                        formatted_chunks.append({
                            "text": text,
                            "score": score,
                            "metadata": meta
                        })

            # Filter by minimum score if applicable
            if min_score is not None and any(c["score"] is not None for c in formatted_chunks):
                if self.score_field.lower() in ["distance", "dist"]:  # lower = better
                    formatted_chunks = [c for c in formatted_chunks if c["score"] is None or c["score"] <= min_score]
                else:  # similarity / cosine (higher = better)
                    formatted_chunks = [c for c in formatted_chunks if c["score"] is None or c["score"] >= min_score]

            retrieved_count = len(formatted_chunks)

            if retrieved_count == 0:
                logger.warning(f"No relevant chunks found for query in doc {doc_id[:8]}")
                return {
                    "status": "warning",
                    "query": query,
                    "chunks": [],
                    "retrieved_count": 0,
                    "message": "No relevant context found in the document for this question.",
                    "raw_results": search_result
                }

            logger.info(f"Retrieval successful | retrieved {retrieved_count} chunks")

            return {
                "status": "success",
                "query": query,
                "chunks": formatted_chunks,          # ← preferred format for LLM agent
                "retrieved_count": retrieved_count,
                "raw_results": search_result,        # useful for debugging
                "doc_id": doc_id
            }

        except Exception as e:
            logger.exception("Retrieval failed")
            return self._error_response(str(e))

    def _error_response(self, msg: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "query": None,
            "chunks": [],
            "retrieved_count": 0,
            "error": msg,
            "message": "Failed to retrieve context from document"
        }

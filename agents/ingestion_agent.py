import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from utils.parser import parse_document
from utils.embedding import get_embeddings
from utils.vector_store import save_embeddings

logger = logging.getLogger(__name__)

class IngestionAgent:
    """
    Handles document ingestion:
      - Parses file into text chunks
      - Creates embeddings using Sentence Transformers (all-MiniLM-L6-v2)
      - Stores chunks + embeddings + metadata in vector store
    """

    def __init__(
        self,
        default_chunk_size: int = 1000,
        default_chunk_overlap: int = 180,
    ):
        self.chunk_size = default_chunk_size
        self.chunk_overlap = default_chunk_overlap

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP-style message handler.

        Expected input message format:
            {
                "payload": {
                    "file_path": str,
                    # optional:
                    "chunk_size": int | None,
                    "chunk_overlap": int | None,
                    "metadata": dict | None
                },
                ...
            }

        Returns:
            {
                "status": "success" | "error",
                "doc_id": str | None,
                "chunk_count": int | None,
                "message": str | None,
                "error": str | None (only on failure)
            }
        """
        payload = message.get("payload", {})
        file_path = payload.get("file_path")

        if not file_path:
            return self._error_response("Missing file_path in payload")

        if not os.path.isfile(file_path):
            return self._error_response(f"File not found: {file_path}")

        try:
            # ── Override defaults if provided ───────────────────────────────
            chunk_size = payload.get("chunk_size", self.chunk_size)
            chunk_overlap = payload.get("chunk_overlap", self.chunk_overlap)
            extra_metadata = payload.get("metadata", {})

            logger.info(f"Ingestion started | file={os.path.basename(file_path)}")

            # 1. Parse document → list of text chunks
            chunks: List[str] = parse_document(
                file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            if not chunks:
                return self._error_response("No text chunks extracted from document")

            logger.info(f"Extracted {len(chunks)} chunks")

            # 2. Generate embeddings using Sentence Transformers
            embeddings: List[List[float]] = get_embeddings(
                chunks=chunks,
                batch_size=32,           # adjust based on your memory
                show_progress=True
            )

            if len(embeddings) != len(chunks):
                return self._error_response(
                    f"Embedding / chunk length mismatch: {len(embeddings)} vs {len(chunks)}"
                )

            # 3. Create unique document identifier
            doc_id = str(uuid.uuid4())

            # 4. Prepare per-chunk metadata
            base_metadata = {
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "source": "ingestion_agent",
                "embedding_model": "all-MiniLM-L6-v2",
                **extra_metadata
            }

            chunk_metadata_list = []
            for i, chunk in enumerate(chunks):
                chunk_meta = base_metadata.copy()
                chunk_meta.update({
                    "chunk_index": i,
                    "chunk_length": len(chunk),
                    # "page_number": ...    # ← add if your parser supports it
                })
                chunk_metadata_list.append(chunk_meta)

            # 5. Store everything
            save_embeddings(
                doc_id=doc_id,
                chunks=chunks,
                embeddings=embeddings,
                metadatas=chunk_metadata_list,
            )

            logger.info(f"Ingestion completed | doc_id={doc_id} | chunks={len(chunks)}")

            return {
                "status": "success",
                "doc_id": doc_id,
                "chunk_count": len(chunks),
                "file_name": os.path.basename(file_path),
                "message": f"Document indexed successfully ({len(chunks)} chunks)"
            }

        except Exception as e:
            logger.exception("Ingestion failed")
            return self._error_response(str(e), doc_id=None)

    def _error_response(self, msg: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        return {
            "status": "error",
            "doc_id": doc_id,
            "chunk_count": None,
            "message": None,
            "error": msg
        }


# Optional: helper to configure logging once at app startup
def configure_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-18s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


# ────────────────────────────────────────────────
#  If you want to test the agent standalone
# ────────────────────────────────────────────────
if __name__ == "__main__":
    configure_logging()

    # Example usage
    agent = IngestionAgent()

    test_message = {
        "payload": {
            "file_path": "data/example.pdf",  # ← change to real test file
            "chunk_size": 800,
            "chunk_overlap": 150,
            "metadata": {"category": "test", "year": 2026}
        }
    }

    result = agent.handle_message(test_message)
    print(result)

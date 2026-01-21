import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from utils.parser import parse_document
from utils.embedding import get_gemini_embedding
from utils.vector_store import save_embeddings
# from utils.mcp import create_message   ← usually not needed inside the agent

logger = logging.getLogger(__name__)

class IngestionAgent:
    """
    Handles document ingestion:
      - Parses file into text chunks
      - Creates embeddings using Gemini
      - Stores chunks + embeddings + metadata in vector store
    """

    def __init__(
        self,
        default_chunk_size: int = 1000,
        default_chunk_overlap: int = 180,
        embedding_model: str = "models/embedding-001",  # common Gemini embedding model
    ):
        self.chunk_size = default_chunk_size
        self.chunk_overlap = default_chunk_overlap
        self.embedding_model = embedding_model

        # You can later make these configurable via environment variables or constructor args

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
                # add more kwargs if your parse_document supports them
            )

            if not chunks:
                return self._error_response("No text chunks extracted from document")

            logger.info(f"Extracted {len(chunks)} chunks")

            # 2. Generate embeddings
            embeddings: List[List[float]] = get_gemini_embedding(
                chunks,
                model=self.embedding_model
            )

            if len(embeddings) != len(chunks):
                return self._error_response(
                    f"Embedding / chunk length mismatch: {len(embeddings)} vs {len(chunks)}"
                )

            # 3. Create unique document identifier
            doc_id = str(uuid.uuid4())

            # 4. Prepare per-chunk metadata (very valuable for retrieval & traceability)
            base_metadata = {
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "source": "ingestion_agent",
                **extra_metadata  # user can pass e.g. {"department": "legal", "year": 2025}
            }

            chunk_metadata_list = []
            for i in range(len(chunks)):
                chunk_meta = base_metadata.copy()
                chunk_meta.update({
                    "chunk_index": i,
                    "chunk_length": len(chunks[i]),
                    # Add page numbers, section titles, etc. if your parser supports it
                })
                chunk_metadata_list.append(chunk_meta)

            # 5. Store everything
            save_embeddings(
                doc_id=doc_id,
                chunks=chunks,
                embeddings=embeddings,
                metadatas=chunk_metadata_list,   # ← most vector stores support this
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
            "message": None,
            "error": msg
        }


# Quick helper (optional) – call once at app startup
def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-18s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

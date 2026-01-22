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
    Handles document ingestion with detailed diagnostics.
    """

    def __init__(
        self,
        default_chunk_size: int = 1000,
        default_chunk_overlap: int = 180,
    ):
        self.chunk_size = default_chunk_size
        self.chunk_overlap = default_chunk_overlap

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        payload = message.get("payload", {})
        file_path = payload.get("file_path")

        if not file_path:
            return self._error_response("Missing file_path in payload")

        if not os.path.isfile(file_path):
            return self._error_response(f"File not found: {file_path}")

        try:
            chunk_size = payload.get("chunk_size", self.chunk_size)
            chunk_overlap = payload.get("chunk_overlap", self.chunk_overlap)
            extra_metadata = payload.get("metadata", {})

            print(f"[Ingestion] Starting | file={os.path.basename(file_path)}")

            # 1. Parse
            print("[Ingestion] Parsing document...")
            chunks: List[str] = parse_document(
                file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            print(f"[Ingestion] Extracted {len(chunks)} chunks")

            if not chunks:
                return self._error_response("No text chunks extracted from document")

            # 2. Embed
            print("[Ingestion] Generating embeddings...")
            embeddings: List[List[float]] = get_embeddings(
                chunks=chunks,
                batch_size=32,
                show_progress=True
            )
            print(f"[Ingestion] Created {len(embeddings)} embeddings")

            if len(embeddings) != len(chunks):
                return self._error_response(
                    f"Embedding/chunk mismatch: {len(embeddings)} vs {len(chunks)}"
                )

            # 3. Doc ID
            doc_id = str(uuid.uuid4())
            print(f"[Ingestion] Generated doc_id = {doc_id}")

            # 4. Metadata
            print("[Ingestion] Preparing metadata...")
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
                })
                chunk_metadata_list.append(chunk_meta)
            print(f"[Ingestion] Created {len(chunk_metadata_list)} metadata entries")

            # 5. Save – this is the most suspicious part
            print("[Ingestion] Saving to vector store...")
            try:
                save_embeddings(
                    doc_id=doc_id,
                    chunks=chunks,
                    embeddings=embeddings,
                    metadatas=chunk_metadata_list,
                )
                print("[Ingestion] save_embeddings completed successfully")
            except Exception as save_exc:
                print(f"[Ingestion] ERROR in save_embeddings: {type(save_exc).__name__}: {save_exc}")
                logger.exception("save_embeddings failed")
                return self._error_response(f"Vector store save failed: {str(save_exc)}", doc_id=doc_id)

            # Success
            print("[Ingestion] Ingestion completed successfully")
            return {
                "status": "success",
                "doc_id": doc_id,
                "chunk_count": len(chunks),
                "file_name": os.path.basename(file_path),
                "message": f"Document indexed successfully ({len(chunks)} chunks)"
            }

        except Exception as e:
            print(f"[Ingestion] GENERAL EXCEPTION: {type(e).__name__}: {str(e)}")
            logger.exception("Ingestion failed")
            return self._error_response(str(e), doc_id=None)

    def _error_response(self, msg: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        print(f"[Ingestion] Returning ERROR: {msg}")
        return {
            "status": "error",
            "doc_id": doc_id,
            "chunk_count": None,
            "message": None,
            "error": msg
        }


# Optional logging config
def configure_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-18s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


if __name__ == "__main__":
    configure_logging()
    agent = IngestionAgent()
    test_message = {
        "payload": {
            "file_path": "data/Kilambi Harivadan.pdf",
            "chunk_size": 800,
            "chunk_overlap": 150,
        }
    }
    result = agent.handle_message(test_message)
    print("Final result:", result)

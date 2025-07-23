import os
from sentence_transformers import SentenceTransformer
from utils.parser import parse_document
from utils.embedding import get_gemini_embedding
from utils.vector_store import save_embeddings
from utils.mcp import create_message


class IngestionAgent:
    def __init__(self):
        # Force use of CPU to avoid Streamlit Cloud GPU issues
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def handle_message(self, message):
        """
        Handles the INGEST message to process and embed the document.
        Message must include file_path in payload.
        """
        if message["type"] != "INGEST":
            return create_message(
                sender="IngestionAgent",
                receiver=message["sender"],
                msg_type="ERROR",
                trace_id=message["trace_id"],
                payload={"error": "Unsupported message type"}
            )

        file_path = message["payload"].get("file_path")
        if not file_path or not os.path.exists(file_path):
            return create_message(
                sender="IngestionAgent",
                receiver=message["sender"],
                msg_type="ERROR",
                trace_id=message["trace_id"],
                payload={"error": f"File not found: {file_path}"}
            )

        # Parse content from uploaded document
        chunks = parse_document(file_path)

        # Get embeddings using Gemini (or fallback to local model if needed)
        embeddings = [get_gemini_embedding(chunk) for chunk in chunks]

        # Save embeddings to vector store (e.g., FAISS)
        save_embeddings(embeddings, chunks)

        return create_message(
            sender="IngestionAgent",
            receiver=message["sender"],
            msg_type="READY",
            trace_id=message["trace_id"],
            payload={"status": "Ingestion complete", "chunks": len(chunks)}
        )

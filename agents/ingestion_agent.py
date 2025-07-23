import os
import pickle
from sentence_transformers import SentenceTransformer
from utils.parser import parse_document
from utils.mcp import create_message, parse_message

class IngestionAgent:
    def __init__(self, name="IngestionAgent", storage_path="vector_store.pkl"):
        self.name = name
        self.storage_path = storage_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = parse_message(message)

        file_path = payload["file_path"]

        # Parse and chunk the document
        chunks = parse_document(file_path)
        if not chunks:
            raise ValueError("No text could be extracted from the document.")

        # Embed chunks
        embeddings = self.model.encode(chunks, convert_to_tensor=True)

        # Save to vector store
        with open(self.storage_path, "wb") as f:
            pickle.dump((chunks, embeddings), f)

        # Create response message
        response_payload = {"status": "success", "chunks_stored": len(chunks)}
        return create_message(self.name, sender, "INGESTION_RESPONSE", trace_id, response_payload)


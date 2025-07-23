import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from utils.parser import parse_document
from utils.mcp import create_message, parse_message

class IngestionAgent:
    def __init__(self, name="IngestionAgent", storage_path="vector_store.pkl"):
        self.name = name
        self.storage_path = storage_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # Force CPU

    def embed_text(self, texts):
        return self.model.encode(texts, convert_to_tensor=False)

    def save_embeddings(self, texts, embeddings):
        with open(self.storage_path, "wb") as f:
            pickle.dump((texts, embeddings), f)

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = parse_message(message)

        if msg_type != "INGEST":
            raise ValueError("Unsupported message type")

        file_path = payload["file_path"]
        texts = parse_document(file_path)
        embeddings = self.embed_text(texts)
        self.save_embeddings(texts, embeddings)

        return create_message(
            sender=self.name,
            receiver=sender,
            msg_type="INGESTED",
            trace_id=trace_id,
            payload={"status": "success", "num_chunks": len(texts)}
        )


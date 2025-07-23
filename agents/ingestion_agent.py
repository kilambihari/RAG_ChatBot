from sentence_transformers import SentenceTransformer
from utils.parser import parse_document
from utils.mcp import create_message, parse_message
import os
import pickle
import uuid
from sklearn.metrics.pairwise import cosine_similarity

class IngestionAgent:
    def __init__(self):
        # Just load the model normally (don't set device)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = parse_message(message)

        file_path = payload["file_path"]
        raw_text = parse_document(file_path)

        chunks = [raw_text[i:i+500] for i in range(0, len(raw_text), 500)]
        embeddings = self.model.encode(chunks).tolist()

        vector_store = list(zip(chunks, embeddings))
        with open("vector_store.pkl", "wb") as f:
            pickle.dump(vector_store, f)

        return create_message(
            sender=receiver,
            receiver=sender,
            msg_type="INGESTION_COMPLETE",
            trace_id=trace_id,
            payload={"status": "completed", "chunks": len(chunks)},
        )

import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from utils.mcp import create_message, parse_message

class RetrievalAgent:
    def __init__(self, name="RetrievalAgent", storage_path="vector_store.pkl", top_k=3):
        self.name = name
        self.storage_path = storage_path
        self.top_k = top_k
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # Force CPU
        self.texts = []
        self.embeddings = None

    def load_embeddings(self):
        if not os.path.exists(self.storage_path):
            raise FileNotFoundError("Vector store not found. Please upload a document first.")
        with open(self.storage_path, "rb") as f:
            self.texts, self.embeddings = pickle.load(f)

    def retrieve(self, query):
        query_embedding = self.model.encode([query])[0]
        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[-self.top_k:][::-1]
        return [self.texts[i] for i in top_indices]

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = parse_message(message)

        if msg_type != "RETRIEVE":
            raise ValueError("Unsupported message type")

        query = payload["query"]
        self.load_embeddings()
        results = self.retrieve(query)

        return create_message(
            sender=self.name,
            receiver=sender,
            msg_type="RETRIEVED",
            trace_id=trace_id,
            payload={"results": results}
        )

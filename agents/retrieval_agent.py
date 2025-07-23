import os
import pickle
import torch
from sentence_transformers import SentenceTransformer, util
from utils.mcp import create_message, parse_message

class RetrievalAgent:
    def __init__(self, name="RetrievalAgent", storage_path="vector_store.pkl", top_k=3):
        self.name = name
        self.storage_path = storage_path
        self.top_k = top_k
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks = []
        self.embeddings = None

    def load_embeddings(self):
        if not os.path.exists(self.storage_path):
            raise FileNotFoundError("Vector store not found. Please upload a document first.")

        with open(self.storage_path, "rb") as f:
            self.chunks, self.embeddings = pickle.load(f)

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = parse_message(message)

        query = payload["query"]
        self.load_embeddings()

        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.embeddings)[0]
        top_results = torch.topk(scores, k=min(self.top_k, len(self.chunks)))

        retrieved_chunks = [self.chunks[idx] for idx in top_results.indices]

        response_payload = {"retrieved_chunks": retrieved_chunks}
        return create_message(self.name, sender, "RETRIEVAL_RESPONSE", trace_id, response_payload)

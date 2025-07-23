# agents/retrieval_agent.py

import os
import pickle
import torch
from sentence_transformers import SentenceTransformer, util

from utils.mcp import parse_message, create_message

class RetrievalAgent:
    def __init__(self):
        self.name = "RetrievalAgent"
        self.embeddings = None
        self.chunks = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_embeddings(self, filename="vectorstore.pkl"):
        if not os.path.exists(filename):
            raise FileNotFoundError("Vector store not found. Please upload a document first.")
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.embeddings = torch.tensor(data["embeddings"])
            self.chunks = data["chunks"]

    def retrieve_top_k_chunks(self, query, k=5):
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.embeddings)[0]
        top_k = torch.topk(scores, k=k)
        top_chunks = [self.chunks[idx] for idx in top_k.indices]
        return top_chunks

    def handle_message(self, message: dict) -> dict:
        sender, receiver, msg_type, trace_id, payload = parse_message(message)
        
        self.load_embeddings()
        query = payload.get("query")
        top_chunks = self.retrieve_top_k_chunks(query)

        response_payload = {"chunks": top_chunks}
        return create_message(
            sender=self.name,
            receiver=sender,
            msg_type="retrieval_result",
            trace_id=trace_id,
            payload=response_payload
        )


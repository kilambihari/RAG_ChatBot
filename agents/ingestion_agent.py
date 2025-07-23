from transformers import AutoTokenizer, AutoModel
import torch
from utils.parser import parse_document
from utils.mcp import create_message, parse_message
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class IngestionAgent:
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element is the token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_embeddings(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings.numpy().tolist()

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = parse_message(message)
        file_path = payload["file_path"]

        raw_text = parse_document(file_path)
        chunks = [raw_text[i:i+500] for i in range(0, len(raw_text), 500)]

        embeddings = self.get_embeddings(chunks)
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

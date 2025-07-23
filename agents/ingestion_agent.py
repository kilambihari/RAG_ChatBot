import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from utils.parser import parse_file
from utils.mcp import create_message, parse_message


class IngestionAgent:
    def __init__(self):
        self.name = "IngestionAgent"

        # Load tokenizer and model from Hugging Face (CPU-safe)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element: token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, texts):
        # Sanitize text chunks
        cleaned_texts = [str(t).strip() for t in texts if isinstance(t, str) and t.strip()]
        if not cleaned_texts:
            raise ValueError("No valid text chunks to embed.")

        encoded_input = self.tokenizer(cleaned_texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings.numpy().tolist()

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = parse_message(message)
        file_path = payload["file_path"]

        # Parse the document
        raw_text = parse_file(file_path)

        # Split into chunks for embedding
        chunks = [raw_text[i:i+500] for i in range(0, len(raw_text), 500) if raw_text[i:i+500].strip()]

        # Generate embeddings
        embeddings = self.get_embeddings(chunks)

        # Return new message
        return create_message(
            sender=self.name,
            receiver="RetrievalAgent",
            msg_type="ingestion_result",
            trace_id=trace_id,
            payload={"chunks": chunks, "embeddings": embeddings}
        )

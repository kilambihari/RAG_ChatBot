import os
from sentence_transformers import SentenceTransformer
from utils.parser import parse_file
from utils.file_handler import chunk_text
from utils.mcp import create_message, parse_message

class IngestionAgent:
    def __init__(self):
        self.name = "IngestionAgent"
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_embeddings(self, chunks):
        # Convert list of strings into sentence embeddings
        return self.model.encode(chunks, show_progress_bar=True).tolist()

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = parse_message(message)

        file_path = payload.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise ValueError(f"Invalid or missing file: {file_path}")

        # Parse file and chunk it
        text = parse_file(file_path)
        chunks = chunk_text(text)
        embeddings = self.get_embeddings(chunks)

        # Return structured message
        return create_message(
            sender=self.name,
            receiver="RetrievalAgent",
            msg_type="INGESTION_RESULT",
            trace_id=trace_id,
            payload={
                "chunks": chunks,
                "embeddings": embeddings,
                "file_path": file_path
            }
        )

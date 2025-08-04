from utils.parser import parse_document
from utils.embedding import get_gemini_embedding
from utils.vector_store import save_embeddings
from utils.mcp import create_message
import uuid

class IngestionAgent:
    def __init__(self):
        pass

    def handle_message(self, message):
        file_path = message["payload"]["file_path"]
        content_chunks = parse_document(file_path)
        embeddings = get_gemini_embedding(content_chunks)
        doc_id = str(uuid.uuid4())
        save_embeddings(doc_id, content_chunks, embeddings)
        return {"doc_id": doc_id}

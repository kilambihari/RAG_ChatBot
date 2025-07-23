import os
import faiss
import numpy as np
import pickle

from utils.mcp import create_message, parse_message

class RetrievalAgent:
    def __init__(self):
        self.name = "RetrievalAgent"
        self.index = None
        self.chunks = []
        self.file_path = ""
        self.index_path = "vector_store/index.faiss"
        self.meta_path = "vector_store/meta.pkl"

    def build_vector_store(self, embeddings):
        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float32"))

    def save_vector_store(self, embeddings, chunks, file_path):
        os.makedirs("vector_store", exist_ok=True)
        self.build_vector_store(embeddings)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump({"chunks": chunks, "file_path": file_path}, f)

    def load_vector_store(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("Vector store not found.")
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.file_path = data["file_path"]

    def search(self, query_embedding, top_k=5):
        if self.index is None:
            self.load_vector_store()
        D, I = self.index.search(np.array([query_embedding]).astype("float32"), top_k)
        return [self.chunks[i] for i in I[0]]

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = parse_message(message)

        if msg_type == "INGESTION_RESULT":
            chunks = payload.get("chunks", [])
            embeddings = payload.get("embeddings", [])
            file_path = payload.get("file_path", "")
            self.chunks = chunks
            self.file_path = file_path
            self.save_vector_store(embeddings, chunks, file_path)

            return create_message(
                sender=self.name,
                receiver="LLMResponseAgent",
                msg_type="READY",
                trace_id=trace_id,
                payload={"file_path": file_path}
            )

        elif msg_type == "QUERY":
            query_embedding = payload.get("query_embedding")
            if query_embedding is None:
                raise ValueError("Missing query embedding in payload")
            top_chunks = self.search(query_embedding)
            return create_message(
                sender=self.name,
                receiver="LLMResponseAgent",
                msg_type="RETRIEVAL_RESULT",
                trace_id=trace_id,
                payload={"top_chunks": top_chunks}
            )

        else:
            raise ValueError(f"Unsupported message type: {msg_type}")

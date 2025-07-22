from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RetrievalAgent:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embed_model = SentenceTransformer(model_name)
        self.index = None
        self.text_chunks = []

    def build_index(self, chunks):
        self.text_chunks = chunks
        embeddings = self.embed_model.encode(chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def retrieve(self, query, top_k=5):
        query_embedding = self.embed_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)
        return [self.text_chunks[i] for i in indices[0]]

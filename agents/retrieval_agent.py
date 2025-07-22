

from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
import torch
from langchain.docstore.document import Document

class RetrievalAgent:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        device = "cpu"  # Force CPU
        self.embed_model = SentenceTransformer(model_name)
        self.embed_model.to(torch.device(device))
        self.vector_store = None

    def build_index(self, chunks):
        docs = [Document(page_content=c) for c in chunks]
        self.vector_store = FAISS.from_documents(docs, self.embed_model)

    def retrieve(self, query, top_k=5):
        if not self.vector_store:
            return ["Index not built yet."]
        results = self.vector_store.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]


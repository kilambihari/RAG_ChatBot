from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

class RetrievalAgent:
    def __init__(self):
        self.embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = None
        self.text_chunks = []

    def build_index(self, chunks):
        self.text_chunks = chunks
        self.vector_store = FAISS.from_texts(chunks, self.embedding)

    def retrieve(self, query, top_k=5):
        docs = self.vector_store.similarity_search(query, k=top_k)
        return [doc.page_content for doc in docs]


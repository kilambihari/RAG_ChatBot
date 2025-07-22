from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class RetrievalAgent:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # ✅ Safe initialization on Streamlit Cloud (CPU only, no .to())
        self.embed_model = SentenceTransformer(model_name)

        # Optional: wrap into LangChain-compatible embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # Init FAISS vector store placeholder
        self.vector_store = None

    def store(self, texts: list[str]):
        # Convert texts to LangChain Documents
        docs = [Document(page_content=txt) for txt in texts]

        # Split texts into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)

    def retrieve(self, query: str, k: int = 3):
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)


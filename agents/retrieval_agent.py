import os
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# You can set this env variable locally or use Streamlit secrets if on Streamlit Cloud
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

class RetrievalAgent:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Initialize embeddings using HuggingFace Inference API
        self.embeddings = HuggingFaceHubEmbeddings(
            repo_id=model_name,
            huggingfacehub_api_token=HUGGINGFACE_TOKEN
        )
        self.vector_store = None  # Will be initialized later

    def store(self, texts: list[str]):
        # Step 1: Wrap raw texts into Document objects
        docs = [Document(page_content=text) for text in texts]

        # Step 2: Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        # Step 3: Extract raw text and metadata
        texts_only = [doc.page_content for doc in split_docs]
        metadatas = [doc.metadata for doc in split_docs]

        # Step 4: Embed the text and store in FAISS
        self.vector_store = FAISS.from_texts(texts_only, self.embeddings, metadatas=metadatas)

    def retrieve(self, query: str, k: int = 3):
        if not self.vector_store:
            return "⚠️ Vector store is empty. Please upload documents first."

        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def handle_message(self, message: dict):
        # Expecting: { "type": "store" or "query", "data": [...] or "query": "..." }
        if message["type"] == "store":
            self.store(message["data"])
            return "✅ Documents stored successfully."
        elif message["type"] == "query":
            return self.retrieve(message["query"])
        else:
            return "❌ Unknown message type."


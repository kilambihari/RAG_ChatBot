from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

class RetrievalAgent:
    def __init__(self):
        # Uses hosted Hugging Face inference endpoint (safe for Streamlit Cloud)
        self.embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=st.secrets["HUGGINGFACEHUB_API_TOKEN"],
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None

    def store(self, texts: list[str]):
        # Convert to LangChain Documents
        docs = [Document(page_content=txt) for txt in texts]

        # Split into smaller chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        # Embed and store in FAISS
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)

    def retrieve(self, query: str, k: int = 3):
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)

    def handle_message(self, message: dict) -> dict:
        message_type = message.get("type")

        if message_type == "store":
            texts = message.get("data", [])
            self.store(texts)
            return {"type": "store_result", "data": "✅ Stored chunks successfully."}

        elif message_type == "retrieve":
            query = message.get("data", "")
            results = self.retrieve(query)
            return {"type": "retrieve_result", "data": results}

        else:
            return {"type": "error", "data": f"❌ Unknown message type: {message_type}"}


import os
import re
import streamlit as st
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def clean_text(text: str) -> str:
    """Normalize text for better embedding and matching."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class RetrievalAgent:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        if not token:
            raise ValueError("❌ Hugging Face API token not found in Streamlit secrets!")

        self.embeddings = HuggingFaceHubEmbeddings(
            repo_id=model_name,
            huggingfacehub_api_token=token
        )
        self.vector_store = None

    def store(self, texts: list[str]):
        docs = [Document(page_content=clean_text(txt)) for txt in texts]
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        texts_only = [doc.page_content for doc in split_docs]
        metadatas = [doc.metadata for doc in split_docs]

        self.vector_store = FAISS.from_texts(texts_only, self.embeddings, metadatas=metadatas)

    def retrieve(self, query: str, k: int = 3):
        if not self.vector_store:
            return ["⚠️ No documents have been stored yet."]
        
        cleaned_query = clean_text(query)

        # Improve matching for short queries
        if len(cleaned_query.split()) <= 4 and "who" in cleaned_query:
            cleaned_query += " summary or description"

        docs = self.vector_store.similarity_search(cleaned_query, k=k)
        return [doc.page_content for doc in docs]

    def handle_message(self, message: dict) -> dict:
        action = message.get("action")
        payload = message.get("payload", {})

        if action == "store":
            chunks = payload.get("chunks", [])
            self.store(chunks)
            return {"status": "success", "payload": {"message": "Stored"}}

        elif action == "retrieve":
            query = payload.get("query", "")
            matches = self.retrieve(query)
            return {"status": "success", "payload": {"matches": matches}}

        else:
            return {"status": "error", "payload": {"message": "Unknown action"}}


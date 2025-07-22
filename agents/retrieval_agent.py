from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st


class RetrievalAgent:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceHubEmbeddings(
            repo_id=model_name,
            huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
        )
        self.vector_store = None

    def store(self, texts: list[str]):
        docs = [Document(page_content=txt) for txt in texts]
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        # ✅ Manually extract text content
        texts_only = [doc.page_content for doc in split_docs]
        metadatas = [doc.metadata for doc in split_docs]

    # ✅ Use from_texts instead of from_documents
    self.vector_store = FAISS.from_texts(texts_only, self.embeddings, metadatas=metadatas)

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

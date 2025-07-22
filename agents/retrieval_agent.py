import os
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class RetrievalAgent:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise ValueError("❌ Hugging Face API token not found in environment!")

        self.embeddings = HuggingFaceHubEmbeddings(
            repo_id=model_name,
            huggingfacehub_api_token=token
        )
        self.vector_store = None

    def store(self, texts: list[str]):
        docs = [Document(page_content=txt) for txt in texts]
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)

        texts_only = [doc.page_content for doc in split_docs]
        metadatas = [doc.metadata for doc in split_docs]

        self.vector_store = FAISS.from_texts(texts_only, self.embeddings, metadatas=metadatas)

    def retrieve(self, query: str, k: int = 3):
        if not self.vector_store:
            return ["⚠️ No documents have been stored yet."]
        docs = self.vector_store.similarity_search(query, k=k)
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


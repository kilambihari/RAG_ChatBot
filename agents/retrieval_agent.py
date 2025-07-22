from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RetrievalAgent:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # ✅ No SentenceTransformer (avoids .to(device))
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None

    def store(self, texts: list[str]):
        if not texts:
            raise ValueError("❌ No texts provided for storage.")

        split_docs = self.text_splitter.create_documents(texts)
        if not split_docs:
            raise ValueError("❌ Text splitter returned no documents.")

        embeddings = self.embeddings.embed_documents([doc.page_content for doc in split_docs])
        if not embeddings:
            raise ValueError("❌ Embedding generation failed. Check model or input format.")

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



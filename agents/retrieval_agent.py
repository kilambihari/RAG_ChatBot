# agents/retrieval_agent.py

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.documents import Document

class RetrievalAgent(Runnable):
    def __init__(self):
        self.vectorstore = None

    def _create_vectorstore(self, chunks: list[Document]):
        # 👇 Force CPU to avoid NotImplementedError in non-GPU environments
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        self.vectorstore = FAISS.from_documents(chunks, embeddings)

    def _retrieve_relevant_chunks(self, query: str) -> list[Document]:
        if not self.vectorstore:
            return []
        docs = self.vectorstore.similarity_search(query, k=4)
        return docs

    def handle_message(self, message: BaseMessage) -> str:
        query = message.content
        relevant_chunks = self._retrieve_relevant_chunks(query)
        if not relevant_chunks:
            return "❌ Sorry, I couldn’t find any relevant information in the document."
        return "\n\n".join([doc.page_content for doc in relevant_chunks])

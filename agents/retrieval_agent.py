from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

class RetrievalAgent:
    def __init__(self):
        self.vectorstore = None

    def _create_vectorstore(self, chunks: list[Document]):
        # Use CPU embeddings from HuggingFace
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

    def handle_message(self, message: dict) -> str:
        if message["action"] == "store":
            self._create_vectorstore(message["data"])
            return "✅ Document stored successfully in vector DB."
        elif message["action"] == "query":
            query = message["data"]
            relevant_chunks = self._retrieve_relevant_chunks(query)
            if not relevant_chunks:
                return "❌ Sorry, no relevant information found in the document."
            return "\n\n".join([doc.page_content for doc in relevant_chunks])
        else:
            return "❌ Unknown message action."


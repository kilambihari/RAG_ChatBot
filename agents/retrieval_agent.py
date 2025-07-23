from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from utils.mcp import create_message

class RetrievalAgent:
    def __init__(self):
        self.vectorstore = None

    def _create_vectorstore(self, chunks: list[Document]):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        self.vectorstore = FAISS.from_documents(chunks, embeddings)

    def _retrieve_relevant_chunks(self, query: str) -> list[Document]:
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search(query, k=4)

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = message.values()

        if msg_type == "DOCUMENT_CHUNKS":
            chunks = payload["chunks"]
            self._create_vectorstore(chunks)
            return create_message(
                sender="RetrievalAgent",
                receiver="UI",
                msg_type="VECTOR_STORE_OK",
                trace_id=trace_id,
                payload={"status": "stored"}
            )

        elif msg_type == "QUERY":
            query = payload["query"]
            relevant_chunks = self._retrieve_relevant_chunks(query)
            context = [doc.page_content for doc in relevant_chunks]

            return create_message(
                sender="RetrievalAgent",
                receiver="LLMResponseAgent",
                msg_type="RETRIEVAL_RESULT",
                trace_id=trace_id,
                payload={"retrieved_context": context, "query": query}
            )


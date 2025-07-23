from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from utils.mcp import create_message
import uuid

class RetrievalAgent:
    def __init__(self, agent_id="RetrievalAgent"):
        self.agent_id = agent_id
        self.vectorstore = None

    def _create_vectorstore(self, chunks):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    def handle_message(self, message: dict) -> dict:
        sender = message.get("from")
        receiver = message.get("to")
        msg_type = message.get("type")
        trace_id = message.get("trace_id", None)
        payload = message.get("payload", {})

        chunks = payload.get("chunks")
        if not chunks:
            raise ValueError("No chunks found in payload")

        self._create_vectorstore(chunks)

        return create_message(
            sender=self.agent_id,
            receiver=sender,
            msg_type="RETRIEVAL_READY",
            trace_id=trace_id,
            payload={"status": "Vectorstore created"}
        )

    def retrieve(self, query, k=3):
        if not self.vectorstore:
            raise ValueError("Vectorstore is not initialized.")
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

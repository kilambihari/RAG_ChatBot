# agents/retrieval_agent.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from typing import List, Dict, Any
import os


class RetrievalAgent:
    def __init__(self):
        self.vectorstore = None

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles a message from another agent using MCP-style message structure.
        Expects a message with type 'document_chunks' and payload containing 'chunks'.
        """
        if message.get("type") != "document_chunks":
            return {
                "type": "error",
                "from": "RetrievalAgent",
                "to": message.get("from", "Unknown"),
                "payload": {"error": "Unsupported message type"}
            }

        chunks = message["payload"].get("chunks", [])
        self._create_vectorstore(chunks)

        return {
            "type": "vectorstore_ready",
            "from": "RetrievalAgent",
            "to": message["from"],
            "payload": {"status": "Vectorstore created from chunks"}
        }

    def _create_vectorstore(self, chunks: List[str]):
        """
        Creates a FAISS vectorstore using HuggingFace sentence-transformer embeddings.
        Forces CPU usage for compatibility on limited environments.
        """
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}  # 👈 Explicitly force CPU to avoid NotImplementedError
        )

        self.vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    def query(self, user_query: str) -> List[Document]:
        """
        Queries the FAISS vectorstore for similar documents.
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")

        results = self.vectorstore.similarity_search(user_query, k=5)
        return results

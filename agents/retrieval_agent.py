from sentence_transformers import SentenceTransformer, util
from utils.mcp import MCPMessage
import torch

class RetrievalAgent:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = []
        self.embeddings = []

    def handle_message(self, message: MCPMessage):
        if message.type == "store_chunks":
            self.documents = message.payload["chunks"]
            self.embeddings = [
                self.model.encode(chunk, convert_to_tensor=True)
                for chunk in self.documents
            ]
            return MCPMessage(
                sender="RetrievalAgent",
                type="chunks_stored",
                payload={"status": "stored", "num_chunks": len(self.documents)}
            )

        elif message.type == "query":
            query = message.payload["query"]
            if not self.embeddings:
                return MCPMessage(
                    sender="RetrievalAgent",
                    type="error",
                    payload={"message": "No documents stored for retrieval."}
                )

            query_emb = self.model.encode(query, convert_to_tensor=True)

            # Ensure embeddings are stacked into a tensor (fixes the crash)
            embeddings_tensor = torch.stack(self.embeddings)

            scores = util.cos_sim(query_emb, embeddings_tensor)[0]
            top_k = min(3, len(scores))  # handle less than 3 chunks
            top_indices = torch.topk(scores, k=top_k).indices.tolist()
            top_chunks = [self.documents[i] for i in top_indices]

            return MCPMessage(
                sender="RetrievalAgent",
                type="retrieved_chunks",
                payload={"chunks": top_chunks}
            )

        else:
            return MCPMessage(
                sender="RetrievalAgent",
                type="error",
                payload={"message": f"Unknown message type: {message.type}"}
            )



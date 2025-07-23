# agents/retrieval_agent.py

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
            self.embeddings = []

            for chunk in self.documents:
                emb = self.model.encode(chunk, convert_to_tensor=True)
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb)
                self.embeddings.append(emb)

            return MCPMessage(
                sender="RetrievalAgent",
                type="chunks_stored",
                payload={"status": "stored", "num_chunks": len(self.documents)}
            )

        elif message.type == "query":
            query = message.payload["query"]

            # Ensure we have embeddings and documents
            if not self.embeddings or not self.documents:
                return MCPMessage(
                    sender="RetrievalAgent",
                    type="error",
                    payload={"message": "No documents available for retrieval."}
                )

            query_emb = self.model.encode(query, convert_to_tensor=True)
            if not isinstance(query_emb, torch.Tensor):
                query_emb = torch.tensor(query_emb)

            try:
                # Stack all embeddings into a tensor
                stacked_embeddings = torch.stack(self.embeddings)

                # Compute cosine similarity
                scores = util.cos_sim(query_emb, stacked_embeddings)[0]

                # Get top-k chunks
                top_k = min(3, len(scores))
                top_indices = torch.topk(scores, k=top_k).indices.tolist()
                top_chunks = [self.documents[i] for i in top_indices]

                return MCPMessage(
                    sender="RetrievalAgent",
                    type="retrieved_chunks",
                    payload={"chunks": top_chunks}
                )

            except Exception as e:
                return MCPMessage(
                    sender="RetrievalAgent",
                    type="error",
                    payload={"message": f"Similarity error: {str(e)}"}
                )

        else:
            return MCPMessage(
                sender="RetrievalAgent",
                type="error",
                payload={"message": f"Unknown message type: {message.type}"}
            )


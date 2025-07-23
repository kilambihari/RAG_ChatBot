from utils.mcp import parse_message, create_message
from sentence_transformers import SentenceTransformer, util

class RetrievalAgent:
    def __init__(self, name="RetrievalAgent"):
        self.name = name
        self.documents = []
        self.embeddings = []
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = parse_message(message)

        if msg_type == "STORE":
            content = payload["content"]
            self.documents.append(content)
            self.embeddings.append(self.model.encode(content, convert_to_tensor=True))
            return create_message(
                self.name,
                sender,
                "STORED",
                trace_id,
                {"status": "Document stored"}
            )

        elif msg_type == "RETRIEVE":
            query = payload["query"]
            query_emb = self.model.encode(query, convert_to_tensor=True)

            if not self.embeddings:
                return create_message(
                    self.name,
                    sender,
                    "RETRIEVED",
                    trace_id,
                    {"context": ""}
                )

            scores = util.cos_sim(query_emb, self.embeddings)[0]
            best_idx = scores.argmax()
            best_chunk = self.documents[best_idx]

            return create_message(
                self.name,
                sender,
                "RETRIEVED",
                trace_id,
                {"context": best_chunk}
            )

        else:
            return create_message(
                self.name,
                sender,
                "ERROR",
                trace_id,
                {"error": f"Unsupported message type: {msg_type}"}
            )

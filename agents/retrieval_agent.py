from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils.mcp import create_message, parse_message


class RetrievalAgent:
    def __init__(self):
        self.name = "RetrievalAgent"
        self.stored_chunks = []
        self.stored_embeddings = []

    def store_document(self, chunks, embeddings):
        self.stored_chunks = chunks
        self.stored_embeddings = np.array(embeddings)

    def retrieve_relevant_chunk(self, question_embedding):
        if not self.stored_embeddings.any():
            return "No document embeddings stored."

        similarities = cosine_similarity(
            np.array(question_embedding).reshape(1, -1),
            self.stored_embeddings
        )[0]

        best_idx = int(np.argmax(similarities))
        return self.stored_chunks[best_idx]

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = parse_message(message)

        if msg_type == "ingestion_result":
            chunks = payload["chunks"]
            embeddings = payload["embeddings"]
            self.store_document(chunks, embeddings)
            return create_message(
                sender=self.name,
                receiver="LLMResponseAgent",
                msg_type="ready_for_questions",
                trace_id=trace_id,
                payload={"status": "stored"}
            )

        elif msg_type == "query":
            question_embedding = payload["question_embedding"]
            retrieved_chunk = self.retrieve_relevant_chunk(question_embedding)
            return create_message(
                sender=self.name,
                receiver="LLMResponseAgent",
                msg_type="retrieved_context",
                trace_id=trace_id,
                payload={"context": retrieved_chunk}
            )

        else:
            return create_message(
                sender=self.name,
                receiver=sender,
                msg_type="error",
                trace_id=trace_id,
                payload={"error": f"Unknown message type: {msg_type}"}
            )


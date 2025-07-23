import os
from utils.mcp import create_message, parse_message
from utils.embedding import get_gemini_embedding
from utils.gemini_wrapper import generate_gemini_response

class LLMResponseAgent:
    def __init__(self):
        self.name = "LLMResponseAgent"
        self.trace_id = None
        self.query = ""
        self.query_embedding = []
        self.retrieved_chunks = []

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = parse_message(message)
        self.trace_id = trace_id

        if msg_type == "USER_QUERY":
            self.query = payload.get("query", "")
            self.query_embedding = get_gemini_embedding(self.query)
            return create_message(
                sender=self.name,
                receiver="RetrievalAgent",
                msg_type="QUERY",
                trace_id=trace_id,
                payload={"query_embedding": self.query_embedding}
            )

        elif msg_type == "RETRIEVAL_RESULT":
            self.retrieved_chunks = payload.get("top_chunks", [])
            prompt = self.build_prompt()
            response = generate_gemini_response(prompt)

            return create_message(
                sender=self.name,
                receiver="App",
                msg_type="FINAL_ANSWER",
                trace_id=trace_id,
                payload={"response": response}
            )

        elif msg_type == "READY":
            # Acknowledgement from retrieval agent after ingestion
            return create_message(
                sender=self.name,
                receiver="App",
                msg_type="READY",
                trace_id=trace_id,
                payload={"status": "ready"}
            )

        else:
            raise ValueError(f"Unsupported message type: {msg_type}")

    def build_prompt(self):
        context = "\n\n".join(self.retrieved_chunks)
        prompt = f"""You are an expert assistant. Use the following context to answer the question:

Context:
{context}

Question:
{self.query}

Answer:"""
        return prompt

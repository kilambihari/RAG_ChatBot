import os
import google.generativeai as genai
from utils.mcp import create_message

class LLMResponseAgent:
    def __init__(self, model_name="gemini-pro"):
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = message.values()

        if msg_type != "RETRIEVAL_RESULT":
            return create_message(
                sender="LLMResponseAgent",
                receiver="UI",
                msg_type="ERROR",
                trace_id=trace_id,
                payload={"error": "Invalid message type"}
            )

        query = payload["query"]
        context_chunks = payload["retrieved_context"]

        context_text = "\n".join(context_chunks)
        prompt = f"""Use the following context to answer the question.

Context:
{context_text}

Question:
{query}

Answer:"""

        response = self.model.generate_content(prompt)
        return create_message(
            sender="LLMResponseAgent",
            receiver="UI",
            msg_type="FINAL_ANSWER",
            trace_id=trace_id,
            payload={"answer": response.text}
        )

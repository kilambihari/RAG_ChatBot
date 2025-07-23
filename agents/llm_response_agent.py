import google.generativeai as genai
from utils.mcp import create_message, parse_message
import os

class LLMResponseAgent:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-pro")
        self.name = "LLMResponseAgent"

    def generate_answer(self, question, context):
        prompt = f"""
You are a helpful assistant. Use the context below to answer the user's question.
If the context does not contain the answer, say so honestly.

---
Context:
{context}

Question: {question}
Answer:
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error from Gemini: {str(e)}"

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = parse_message(message)

        if msg_type != "query":
            return create_message(self.name, sender, "error", trace_id, {"error": "Invalid message type"})

        question = payload.get("question")
        relevant_chunks = payload.get("relevant_chunks", [])
        context = "\n".join(relevant_chunks)

        answer = self.generate_answer(question, context)

        return create_message(self.name, sender, "response", trace_id, {"answer": answer})

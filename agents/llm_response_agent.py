# agents/llm_response_agent.py

import google.generativeai as genai
from utils.mcp import parse_message, create_message
import streamlit as st

class LLMResponseAgent:
    def __init__(self, name="LLMResponseAgent"):
        self.name = name
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel("gemini-pro")

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = parse_message(message)

        if msg_type == "GENERATE":
            query = payload["query"]
            context = payload.get("context", "")

            prompt = f"""You are a helpful assistant. Use the provided context to answer the user's query.
Context:
{context}

Query:
{query}

Answer:"""

            try:
                response = self.model.generate_content(prompt)
                answer = response.text.strip()
            except Exception as e:
                answer = f"❌ Error: {str(e)}"

            return create_message(
                self.name,
                sender,
                "GENERATED",
                trace_id,
                {"response": answer}
            )

        else:
            return create_message(
                self.name,
                sender,
                "ERROR",
                trace_id,
                {"error": f"Unsupported message type: {msg_type}"}
            )


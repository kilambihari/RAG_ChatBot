import google.generativeai as genai
import streamlit as st

class LLMResponseAgent:
    def handle_message(self, message):
        context_chunks = message["payload"]["top_chunks"]
        query = message["payload"]["query"]
        model = genai.GenerativeModel("gemini-pro")

        prompt = f"Context:\n{context_chunks}\n\nQuestion: {query}\nAnswer:"
        response = model.generate_content(prompt)
        
        return {
            "answer": response.text,
            "source_chunks": context_chunks
        }

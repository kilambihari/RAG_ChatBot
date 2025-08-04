import google.generativeai as genai
import streamlit as st

# ✅ Step 1: Configure API Key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

class LLMResponseAgent:
    def handle_message(self, message):
        context_chunks = message["payload"]["top_chunks"]
        query = message["payload"]["query"]

        model = genai.GenerativeModel("models/gemini-pro")

        prompt = f"Context:\n{context_chunks}\n\nQuestion: {query}\nAnswer:"
        
        try:
            response = model.generate_content(prompt)
            return {
                "answer": response.text,
                "source_chunks": context_chunks
            }
        except Exception as e:
            st.error("❌ Gemini API Error: Check API key, model name, or project access.")
            st.exception(e)
            return {
                "answer": "⚠️ Gemini API Error occurred.",
                "source_chunks": context_chunks
            }

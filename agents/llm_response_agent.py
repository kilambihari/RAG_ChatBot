import google.generativeai as genai
import streamlit as st

# Configure Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

class LLMResponseAgent:
    def handle_message(self, message):
        context_chunks = message["payload"]["top_chunks"]
        query = message["payload"]["query"]

        model = genai.GenerativeModel("gemini-1.5-flash")
        context_text = "\n\n".join(context_chunks)

        prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"

        try:
            response = model.generate_content(prompt)
            answer = response.text if hasattr(response, "text") else str(response)
            return {
                "answer": answer,
                "source_chunks": context_chunks
            }
        except Exception as e:
            st.error("❌ Gemini API Error: Check API key, model name, or project access.")
            st.exception(e)
            return {
                "answer": "⚠️ Gemini API Error occurred.",
                "source_chunks": context_chunks
            }

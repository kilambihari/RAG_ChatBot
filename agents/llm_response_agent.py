import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class LLMResponseAgent:
    def handle_message(self, message):
        query = message["data"].get("query", "")
        context_chunks = message["data"].get("chunks", [])

        context = "\n\n".join(context_chunks)
        prompt = f"""You are an AI assistant using RAG.
Use the following context to answer the user query.

Context:
{context}

User query:
{query}

Answer clearly and concisely based on the given context."""

        try:
            # ✅ Correct model name
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            answer = response.text

            return {
                "answer": answer,
                "source_chunks": context_chunks
            }

        except Exception as e:
            return {
                "answer": "⚠️ Gemini API Error occurred.",
                "error": str(e),
                "source_chunks": context_chunks
            }


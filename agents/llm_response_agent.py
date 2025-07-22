import google.generativeai as genai
import os

class LLMResponseAgent:
    def __init__(self, model_name="gemini-pro"):
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def generate_response(self, query, context_chunks):
        context_text = "\n".join(context_chunks)
        prompt = f"""Use the following context to answer the question.

Context:
{context_text}

Question:
{query}

Answer:"""
        response = self.model.generate_content(prompt)
        return response.text

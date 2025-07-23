from utils.mcp import create_message, parse_message
import google.generativeai as genai

class LLMResponseAgent:
    def __init__(self, name="LLMResponseAgent", model_name="models/gemini-pro"):
        self.name = name
        self.model = genai.GenerativeModel(model_name)

    def handle_message(self, message):
        sender, receiver, msg_type, trace_id, payload = parse_message(message)

        query = payload["query"]
        context = payload["context"]

        prompt = (
            "Use the following context to answer the user's question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )

        response = self.model.generate_content(prompt)
        answer = response.text

        response_payload = {"answer": answer}
        return create_message(self.name, sender, "LLM_RESPONSE", trace_id, response_payload)


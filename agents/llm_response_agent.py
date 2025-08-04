from utils.embedding import query_gemini_llm

class LLMResponseAgent:
    def __init__(self):
        pass

    def handle_message(self, message):
        context_chunks = message["payload"]["top_chunks"]
        query = message["payload"]["query"]

        full_prompt = (
            "Answer the question using the following document context:\n\n"
            + "\n---\n".join(context_chunks)
            + f"\n\nQuestion: {query}"
        )

        answer = query_gemini_llm(full_prompt)

        return {
            "answer": answer,
            "source_chunks": context_chunks
        }

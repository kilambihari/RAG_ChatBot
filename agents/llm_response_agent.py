import os
import logging
from typing import Dict, Any, List, Optional
import google.generativeai as genai

logger = logging.getLogger(__name__)

class LLMResponseAgent:
    """
    Handles final answer generation using Gemini + retrieved context (RAG)
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 40,
    ):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        genai.configure(api_key=api_key)

        self.model_name = model_name
        self.generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "top_p": top_p,
            "top_k": top_k,
        }

        # You can also add safety settings if needed
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected incoming message structure (from RetrievalAgent):
        {
            "query": str,
            "chunks": List[str],              # raw text chunks
            # optional but very useful:
            "source_chunks": List[Dict],      # or List[str] with metadata
            "doc_id": str | None,
            "trace_id": str | None,
            ...
        }

        Returns:
        {
            "answer": str,
            "source_chunks": List[str],
            "status": "success" | "error",
            "error": str | None,
            "used_context_length": int | None,
            "model": str
        }
        """
        try:
            query = message.get("query", "").strip()
            chunks = message.get("chunks", []) or message.get("source_chunks", [])

            if not query:
                return self._error_response("No user query provided")

            if not chunks:
                return self._error_response("No context chunks received – cannot answer")

            # ── Build context ────────────────────────────────────────
            # You can improve this part later (sorting, relevance re-ranking, etc.)
            context_parts = []
            for i, chunk in enumerate(chunks, 1):
                # If chunk is dict with metadata → format nicely
                if isinstance(chunk, dict):
                    text = chunk.get("text", chunk.get("content", ""))
                    meta = chunk.get("metadata", {})
                    source_info = f"[{i}] {meta.get('file_name', 'unknown')} • chunk {meta.get('chunk_index', '?')}"
                    context_parts.append(f"{source_info}\n{text}")
                else:
                    # plain string
                    context_parts.append(f"[chunk {i}]\n{chunk}")

            context_str = "\n\n".join(context_parts)

            # ── System + user prompt ─────────────────────────────────
            system_prompt = """You are a precise, helpful RAG assistant.
Answer ONLY using the provided context.
If the context does not contain enough information to answer, say so clearly.
Use markdown formatting when helpful (lists, bold, code blocks).
Be concise unless the user asks for detailed explanation."""

            user_prompt = f"""Context:
{context_str}

User question:
{query}

Answer:"""

            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            # ── Call Gemini ──────────────────────────────────────────
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
            )

            response = model.generate_content(full_prompt)

            answer_text = response.text.strip()

            # Optional: handle cases where model refuses or returns empty
            if not answer_text or "I cannot" in answer_text[:60].lower():
                answer_text = answer_text or "I'm sorry, but I don't have enough information in the provided context to answer this question."

            return {
                "status": "success",
                "answer": answer_text,
                "source_chunks": chunks,           # return original chunks (with metadata if present)
                "model": self.model_name,
                "used_context_length": len(context_str),
                "token_usage": {
                    "prompt": response.usage_metadata.prompt_token_count if hasattr(response, "usage_metadata") else None,
                    "completion": response.usage_metadata.candidates_token_count if hasattr(response, "usage_metadata") else None,
                }
            }

        except Exception as e:
            logger.exception("LLM generation failed")
            return self._error_response(str(e))

    def _error_response(self, msg: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "answer": "⚠️ Error generating answer",
            "error": msg,
            "source_chunks": [],
        }

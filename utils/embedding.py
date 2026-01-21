from typing import List
import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────
#  Embedding Model (Sentence Transformers)
# ────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    """Loads lightweight all-MiniLM-L6-v2 once and caches it"""
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        st.success("Embedding model loaded (all-MiniLM-L6-v2)")
        return model
    except Exception as e:
        st.error(f"Failed to load SentenceTransformer model: {e}")
        logger.exception("Embedding model load failed")
        raise


def get_embeddings(
    chunks: List[str],
    batch_size: int = 32,
    show_progress: bool = True
) -> List[List[float]]:
    """
    Generate embeddings using SentenceTransformer (local, free, fast).
    
    Args:
        chunks: List of text strings to embed
        batch_size: Controls memory usage (higher = faster but more RAM)
    
    Returns:
        List of embedding vectors (each is list of floats)
    """
    if not chunks:
        return []

    model = get_embedding_model()

    try:
        with st.spinner(f"Generating embeddings for {len(chunks)} chunks..."):
            embeddings = model.encode(
                chunks,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True   # usually good for cosine similarity
            )
            embeddings_list = embeddings.tolist()

        st.caption(f"✅ Created {len(embeddings_list)} embeddings (dim={len(embeddings_list[0])})")
        return embeddings_list

    except Exception as e:
        st.error(f"Embedding generation failed: {str(e)}")
        logger.exception("Embedding failed")
        return []


# ────────────────────────────────────────────────
#  Gemini LLM (only for generation / answering)
# ────────────────────────────────────────────────

@st.cache_data(ttl="10min")  # cache the configured model briefly
def get_gemini_model(model_name: str = "gemini-1.5-flash-latest"):
    """Lazy-load and configure Gemini model"""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in secrets")
        raise ValueError("Missing GEMINI_API_KEY")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {e}")
        raise


def generate_with_gemini(
    prompt: str,
    model_name: str = "gemini-1.5-flash-latest",
    temperature: float = 0.2,
    max_output_tokens: int = 2048
) -> str:
    """
    Generate text response using Gemini.
    Returns plain string or error message.
    """
    try:
        model = get_gemini_model(model_name)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "top_p": 0.95,
            "top_k": 40,
        }

        with st.spinner("Generating answer with Gemini..."):
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            text = response.text.strip()

        if not text:
            return "No response received from Gemini."

        return text

    except Exception as e:
        error_msg = f"Gemini API error: {str(e)}"
        st.error(error_msg)
        logger.exception("Gemini generation failed")
        return error_msg


# ────────────────────────────────────────────────
#  Convenience wrapper (RAG-style prompt)
# ────────────────────────────────────────────────

def rag_generate_answer(
    query: str,
    context_chunks: List[str],
    max_context_chunks: int = 6
) -> str:
    """Simple RAG helper: build prompt + generate answer"""
    if not context_chunks:
        return "No relevant context found in the document."

    # Optional: limit context length
    context_text = "\n\n".join(context_chunks[:max_context_chunks])

    prompt = f"""You are a helpful document Q&A assistant.
Answer the question based **only** on the provided context.
If the context does not contain the answer, say so clearly.

Context:
{context_text}

Question: {query}

Answer concisely and accurately:"""

    return generate_with_gemini(prompt)



from sentence_transformers import SentenceTransformer
import streamlit as st

# Load the model once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_gemini_embedding(chunks):
    """
    Generates embeddings using Sentence Transformers (free & offline)
    Compatible replacement for Gemini embeddings.
    """
    model = load_model()
    try:
        embeddings = model.encode(chunks, convert_to_numpy=True).tolist()
        st.success(f"✅ Generated {len(embeddings)} embeddings successfully using SentenceTransformer.")
        return embeddings
    except Exception as e:
        st.error(f"❌ Embedding generation failed: {e}")
        return []

def query_gemini_llm(prompt):
    """
    Keep your Gemini or OpenAI LLM here for answering questions.
    Only the embedding part is replaced.
    """
    import google.generativeai as genai
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini API failed: {e}"


from sentence_transformers import SentenceTransformer
import streamlit as st
import google.generativeai as genai

# Load sentence transformer once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_gemini_embedding(chunks):
    """
    Uses SentenceTransformer instead of Gemini for embeddings
    """
    model = load_model()
    try:
        embeddings = model.encode(chunks, convert_to_numpy=True).tolist()
        st.info(f"✅ Generated {len(embeddings)} embeddings using SentenceTransformer.")
        return embeddings
    except Exception as e:
        st.error(f"Embedding generation failed: {e}")
        return []

def query_gemini_llm(prompt):
    """
    Use Gemini ONLY for text generation (answers)
    """
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini API failed: {e}"



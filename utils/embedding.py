import google.generativeai as genai
import streamlit as st

genai.configure(api_key=st.secrets["gemini_api_key"])

EMBED_MODEL = "models/embedding-001"

def get_gemini_embedding(chunks):
    embeddings = []
    for text in chunks:
        try:
            response = genai.embed_content(
                model=EMBED_MODEL,
                content=text,
                task_type="retrieval_document",
                title="Doc Chunk"
            )
            embeddings.append(response["embedding"])
        except Exception as e:
            st.warning(f"Embedding failed: {e}")
    return embeddings

def query_gemini_llm(prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini API failed: {e}"

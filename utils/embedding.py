# utils/embedding.py

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st

def get_gemini_embedding(text):
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )
    return embeddings_model.embed_query(text)

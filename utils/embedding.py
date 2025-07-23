# utils/embedding.py

import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_gemini_embedding():
    """
    Returns a Gemini-compatible embedding model using LangChain's wrapper.
    Uses API key stored in st.secrets["GEMINI_API_KEY"]
    """
    api_key = st.secrets["GEMINI_API_KEY"]
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

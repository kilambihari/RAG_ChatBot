from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

def get_gemini_embedding():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

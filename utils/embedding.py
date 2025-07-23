from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_gemini_embedding(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks using Gemini embeddings.

    Args:
        texts (List[str]): A list of text chunks.

    Returns:
        List[List[float]]: Corresponding list of embedding vectors.
    """
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings_model.embed_documents(texts)

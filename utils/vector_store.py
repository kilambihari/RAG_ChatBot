import faiss
import numpy as np
import os
import pickle

VECTOR_STORE_PATH = "data/vector_store/"
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

def save_embeddings(doc_id, chunks, embeddings):
    # Safety checks
    if not embeddings or len(embeddings) == 0:
        raise ValueError(f"No embeddings generated for document '{doc_id}'. Check your ingestion or embedding step.")
    if len(chunks) == 0:
        raise ValueError(f"No text chunks found for document '{doc_id}'. Parsing might have failed.")

    # Ensure embeddings and chunks have same length
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings for document '{doc_id}'."
        )

    # Create and store FAISS index
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))

    file_path = os.path.join(VECTOR_STORE_PATH, f"{doc_id}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump((index, chunks), f)

    print(f"Saved FAISS index for document '{doc_id}' with {len(chunks)} chunks at '{file_path}'")

def search_similar_chunks(doc_id, query):
    from utils.embedding import get_gemini_embedding

    file_path = os.path.join(VECTOR_STORE_PATH, f"{doc_id}.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Vector store file not found for document '{doc_id}'")

    with open(file_path, "rb") as f:
        index, chunks = pickle.load(f)

    query_embed = get_gemini_embedding([query])
    if not query_embed or len(query_embed) == 0:
        raise ValueError("Failed to generate query embedding.")

    query_embed = np.array(query_embed[0], dtype="float32").reshape(1, -1)
    D, I = index.search(query_embed, 3)

    results = [chunks[i] for i in I[0] if i < len(chunks)]
    return results


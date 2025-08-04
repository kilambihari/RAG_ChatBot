import faiss
import numpy as np
import os
import pickle

VECTOR_STORE_PATH = "data/vector_store/"

def save_embeddings(doc_id, chunks, embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))
    with open(os.path.join(VECTOR_STORE_PATH, f"{doc_id}.pkl"), "wb") as f:
        pickle.dump((index, chunks), f)

def search_similar_chunks(doc_id, query):
    from utils.embedding import get_gemini_embedding
    with open(os.path.join(VECTOR_STORE_PATH, f"{doc_id}.pkl"), "rb") as f:
        index, chunks = pickle.load(f)
    query_embed = get_gemini_embedding([query])[0]
    D, I = index.search(np.array([query_embed], dtype="float32"), 3)
    return [chunks[i] for i in I[0]]


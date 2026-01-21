import os
import pickle
import numpy as np
import faiss
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

VECTOR_STORE_DIR = "data/vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)


def save_embeddings(
    doc_id: str,
    chunks: List[str],
    embeddings: List[List[float]],
    normalize: bool = True,
) -> str:
    """
    Save text chunks + FAISS index to disk using pickle.
    
    Args:
        doc_id:      Unique document identifier
        chunks:      List of text strings (chunks)
        embeddings:  List of embedding vectors (same length as chunks)
        normalize:   Whether to normalize vectors before indexing (recommended for cosine-like similarity)
    
    Returns:
        Full path to the saved file
    """
    if not chunks:
        raise ValueError(f"No chunks provided for document '{doc_id}'")
    if not embeddings:
        raise ValueError(f"No embeddings provided for document '{doc_id}'")
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Length mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings "
            f"for document '{doc_id}'"
        )

    # Convert to numpy array
    emb_array = np.array(embeddings, dtype=np.float32)

    if normalize:
        faiss.normalize_L2(emb_array)  # in-place normalization (good for cosine similarity)

    dim = emb_array.shape[1]
    if dim <= 0:
        raise ValueError(f"Invalid embedding dimension: {dim}")

    # Create simple exact L2 index (good for < 100k vectors)
    index = faiss.IndexFlatL2(dim)
    index.add(emb_array)

    file_path = os.path.join(VECTOR_STORE_DIR, f"{doc_id}.faiss.pkl")

    try:
        with open(file_path, "wb") as f:
            pickle.dump({
                "index": index,
                "chunks": chunks,
                "dim": dim,
                "count": len(chunks),
                "normalized": normalize,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Saved FAISS store: {doc_id} | {len(chunks)} chunks | dim={dim} | path={file_path}")
        return file_path

    except Exception as e:
        logger.exception(f"Failed to save vector store for {doc_id}")
        raise RuntimeError(f"Could not save vector store: {str(e)}")


def search_similar_chunks(
    doc_id: str,
    query: str,
    k: int = 5,
    min_distance: float = None,          # optional threshold
) -> List[Tuple[str, float]]:
    """
    Retrieve top-k most similar chunks for a query.
    
    Returns:
        List of (chunk_text, distance) tuples, sorted by similarity (smallest distance first)
    """
    from utils.embedding import get_embeddings   # ← use the same function as ingestion

    file_path = os.path.join(VECTOR_STORE_DIR, f"{doc_id}.faiss.pkl")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No vector store found for doc_id '{doc_id}' at {file_path}")

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        index: faiss.Index = data["index"]
        chunks: List[str] = data["chunks"]
        dim: int = data["dim"]
        normalized: bool = data.get("normalized", False)

        if len(chunks) == 0 or index.ntotal == 0:
            return []

        # Generate query embedding (same model/settings as during ingestion!)
        query_emb_list = get_embeddings([query.strip()])
        if not query_emb_list or len(query_emb_list) == 0:
            raise ValueError("Failed to generate query embedding")

        query_vec = np.array(query_emb_list[0], dtype=np.float32).reshape(1, -1)

        if normalized:
            faiss.normalize_L2(query_vec)

        # Search
        distances, indices = index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            if min_distance is not None and dist > min_distance:
                continue
            results.append((chunks[idx], float(dist)))

        logger.debug(f"Retrieved {len(results)} chunks for query in doc {doc_id[:8]}…")

        return results

    except Exception as e:
        logger.exception(f"Search failed for doc_id {doc_id}")
        raise RuntimeError(f"Vector search failed: {str(e)}")


# Optional: helper to list all stored documents
def list_stored_documents() -> List[str]:
    """Return list of doc_ids that have a stored index"""
    files = [f for f in os.listdir(VECTOR_STORE_DIR) if f.endswith(".faiss.pkl")]
    return [os.path.splitext(f)[0] for f in files]

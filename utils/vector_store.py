import os
import faiss
import pickle
import numpy as np

# Save embeddings to disk
def save_embeddings(embedding_list, texts, save_path="vector_store/index.faiss"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert to numpy array
    vectors = np.array(embedding_list).astype("float32")

    # Create FAISS index
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    # Save FAISS index
    faiss.write_index(index, save_path)

    # Save text mapping
    with open(save_path + ".pkl", "wb") as f:
        pickle.dump(texts, f)


# Load FAISS index and texts
def load_embeddings(save_path="vector_store/index.faiss"):
    index = faiss.read_index(save_path)
    with open(save_path + ".pkl", "rb") as f:
        texts = pickle.load(f)
    return index, texts

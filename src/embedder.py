# src/embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once at module level — don't reload it on every call
# First run downloads ~80MB. After that it's cached locally.
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.")


def embed_texts(texts):
    """
    Convert a list of text strings into a numpy array of embeddings.
    
    Args:
        texts : list of strings e.g. ["Python developer", "SQL skills"]
    
    Returns:
        numpy array of shape (len(texts), 384)
        Each row is a 384-dimensional vector representing one text
    """
    if isinstance(texts, str):
        texts = [texts]  # handle single string input
    
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    return embeddings


def embed_single(text):
    """
    Convenience function: embed just one string, return 1D vector.
    Used for embedding the JD query.
    """
    return embed_texts([text])[0]


# ── Test it standalone ──────────────────────────────────────────────
if __name__ == "__main__":
    texts = [
        "Python data analyst with SQL experience",
        "Machine learning engineer with Python",
        "Chef with 5 years experience"
    ]
    
    vecs = embed_texts(texts)
    print(f"Shape: {vecs.shape}")  # should be (3, 384)
    print(f"First vector (first 5 dims): {vecs[0][:5]}")
    
    # Similarity check: texts[0] and texts[1] should be more similar
    # than texts[0] and texts[2]
    from numpy.linalg import norm
    def cosine(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))
    
    print(f"\nSimilarity (Python analyst vs ML engineer): {cosine(vecs[0], vecs[1]):.3f}")
    print(f"Similarity (Python analyst vs Chef):        {cosine(vecs[0], vecs[2]):.3f}")
    # First should be higher than second
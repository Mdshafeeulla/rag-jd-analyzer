# src/vector_store.py

import numpy as np
from numpy.linalg import norm


def cosine_similarity(vec_a, vec_b):
    """
    Measure how similar two vectors are.
    Returns a float between -1 and 1. 
    1.0 = identical meaning, 0.0 = unrelated
    """
    # Avoid division by zero
    if norm(vec_a) == 0 or norm(vec_b) == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))


class VectorStore:
    """
    A simple in-memory vector database.
    Stores text chunks alongside their embeddings.
    Supports similarity search with no external dependencies.
    """
    
    def __init__(self):
        self.chunks = []       # list of raw text strings
        self.embeddings = []   # list of numpy vectors (one per chunk)
        self.metadata = []     # optional: source, section name, etc.
    
    def add(self, chunks, embeddings, metadata=None):
        """
        Add chunks and their embeddings to the store.
        
        Args:
            chunks     : list of text strings
            embeddings : numpy array of shape (N, 384)
            metadata   : optional list of dicts with extra info
        """
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            self.chunks.append(chunk)
            self.embeddings.append(emb)
            if metadata:
                self.metadata.append(metadata[i])
            else:
                self.metadata.append({"index": len(self.chunks) - 1})
        
        print(f"✓ Vector store now has {len(self.chunks)} chunks")
    
    def search(self, query_embedding, top_k=5):
        """
        Find the top_k most similar chunks to the query.
        
        Args:
            query_embedding : 1D numpy vector (the embedded JD)
            top_k           : how many results to return
        
        Returns:
            List of tuples: (chunk_text, similarity_score)
            Sorted by score descending (best match first)
        """
        if not self.chunks:
            raise ValueError("Vector store is empty. Add chunks first.")
        
        # Calculate similarity between query and every stored chunk
        scores = [
            cosine_similarity(query_embedding, emb)
            for emb in self.embeddings
        ]
        
        # Sort indices by score, highest first
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Return (text, score) tuples for top results
        return [(self.chunks[i], round(scores[i], 4)) for i in top_indices]
    
    def __len__(self):
        return len(self.chunks)


# ── Test it standalone ──────────────────────────────────────────────
if __name__ == "__main__":
    from embedder import embed_texts, embed_single
    
    # Simulate resume chunks
    chunks = [
        "Python developer with 3 years experience in data analysis",
        "Proficient in SQL, Excel, and Power BI for reporting",
        "Medical Representative with pharma sales background",
        "Built automation tools using pandas and numpy",
        "B.Sc in Computer Science, graduated 2022"
    ]
    
    embeddings = embed_texts(chunks)
    
    store = VectorStore()
    store.add(chunks, embeddings)
    
    # Simulate a JD query
    query = "Looking for a data analyst with Python and SQL skills"
    query_vec = embed_single(query)
    
    results = store.search(query_vec, top_k=3)
    print("\nTop 3 matches for the query:")
    for text, score in results:
        print(f"\n  Score: {score:.4f}")
        print(f"  Text: {text}")
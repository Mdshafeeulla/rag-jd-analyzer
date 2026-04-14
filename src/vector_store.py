# src/vector_store.py

import math
import re
from collections import Counter, defaultdict

import numpy as np
from numpy.linalg import norm


def tokenize(text):
    """Tokenize text into lowercase alphanumeric terms."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def cosine_similarity(vec_a, vec_b):
    """
    Measure how similar two vectors are.
    Returns a float between -1 and 1. 
    1.0 = identical meaning, 0.0 = unrelated
    """
    if norm(vec_a) == 0 or norm(vec_b) == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))


class VectorStore:
    """
    A simple in-memory vector database.
    Stores text chunks alongside their embeddings.
    Supports hybrid BM25 + semantic similarity search.
    """
    
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.metadata = []

        # BM25 index data
        self.tokenized_chunks = []
        self.term_freqs = []
        self.doc_freq = defaultdict(int)
        self.doc_lens = []
        self.avgdl = 0.0
        self.num_docs = 0
        self.k1 = 1.5
        self.b = 0.75
    
    def _index_bm25(self, chunk):
        tokens = tokenize(chunk)
        self.tokenized_chunks.append(tokens)
        self.term_freqs.append(Counter(tokens))
        self.doc_lens.append(len(tokens))

        for token in set(tokens):
            self.doc_freq[token] += 1

        self.num_docs = len(self.tokenized_chunks)
        self.avgdl = sum(self.doc_lens) / self.num_docs if self.num_docs else 0.0

    def _idf(self, term):
        df = self.doc_freq.get(term, 0)
        return math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

    def _bm25_score(self, query_text, doc_index):
        if self.num_docs == 0:
            return 0.0

        query_tokens = tokenize(query_text)
        if not query_tokens:
            return 0.0

        score = 0.0
        doc_len = self.doc_lens[doc_index]
        freqs = self.term_freqs[doc_index]

        for term in query_tokens:
            tf = freqs.get(term, 0)
            if tf == 0:
                continue

            idf = self._idf(term)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * tf * (self.k1 + 1) / denom

        return score

    def _normalize_scores(self, scores):
        if not scores:
            return []

        max_score = max(scores)
        if max_score <= 0:
            return [0.0] * len(scores)

        return [score / max_score for score in scores]

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
            self._index_bm25(chunk)

            if metadata:
                self.metadata.append(metadata[i])
            else:
                self.metadata.append({"index": len(self.chunks) - 1})

        print(f"✓ Vector store now has {len(self.chunks)} chunks")
    
    def search(self, query_embedding, top_k=5, query_text=None, semantic_weight=0.7):
        """
        Find the top_k most relevant chunks using hybrid retrieval.
        
        Args:
            query_embedding : 1D numpy vector (the embedded JD)
            top_k           : how many results to return
            query_text      : optional raw query text for BM25 scoring
            semantic_weight : balance between semantic and BM25 scores
                              (0.0 = BM25-only, 1.0 = semantic-only)

        Returns:
            List of tuples: (chunk_text, combined_score)
            Sorted by score descending (best match first)
        """
        if not self.chunks:
            raise ValueError("Vector store is empty. Add chunks first.")

        semantic_scores = [
            cosine_similarity(query_embedding, emb)
            for emb in self.embeddings
        ]

        if query_text:
            bm25_scores = [self._bm25_score(query_text, i) for i in range(len(self.chunks))]
            bm25_scores = self._normalize_scores(bm25_scores)
            combined_scores = [
                semantic_weight * sem + (1 - semantic_weight) * bm25
                for sem, bm25 in zip(semantic_scores, bm25_scores)
            ]
        else:
            combined_scores = semantic_scores

        top_indices = sorted(
            range(len(combined_scores)),
            key=lambda i: combined_scores[i],
            reverse=True
        )[:top_k]

        return [(self.chunks[i], round(combined_scores[i], 4)) for i in top_indices]
    
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
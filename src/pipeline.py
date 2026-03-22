# src/pipeline.py

from src.chunker import chunk_text, chunk_by_sections
from src.embedder import embed_texts, embed_single
from src.vector_store import VectorStore
from src.prompt_builder import build_analysis_prompt
from src.llm import ask_ollama


def run_pipeline(resume_text, jd_text, model="mistral", top_k=5):
    """
    Full RAG pipeline: resume → analysis report.
    
    Pipeline steps:
    1. CHUNK   : split resume into pieces
    2. EMBED   : convert chunks to vectors
    3. INDEX   : store vectors in VectorStore
    4. QUERY   : embed the JD
    5. RETRIEVE: find top_k most relevant resume chunks
    6. AUGMENT : build prompt with retrieved context
    7. GENERATE: send to LLM, get analysis
    
    Returns:
        dict with keys: answer, retrieved_chunks, num_chunks
    """
    
    print("\n[1/5] Chunking resume...")
    # Try section-based chunking first, fall back to word-based
    chunks = chunk_by_sections(resume_text)
    if len(chunks) < 3:
        chunks = chunk_text(resume_text, chunk_size=200, overlap=40)
    print(f"      → {len(chunks)} chunks created")
    
    print("[2/5] Embedding resume chunks...")
    chunk_embeddings = embed_texts(chunks)
    print(f"      → Embeddings shape: {chunk_embeddings.shape}")
    
    print("[3/5] Building vector store...")
    store = VectorStore()
    store.add(chunks, chunk_embeddings)
    
    print("[4/5] Retrieving relevant sections for JD...")
    jd_embedding = embed_single(jd_text)
    retrieved = store.search(jd_embedding, top_k=top_k)
    print(f"      → Top {top_k} chunks retrieved (scores: "
          f"{[s for _, s in retrieved]})")
    
    print("[5/5] Generating analysis with LLM...")
    prompt = build_analysis_prompt(retrieved, jd_text)
    answer = ask_ollama(prompt, model=model)
    
    return {
        "answer": answer,
        "retrieved_chunks": retrieved,
        "num_chunks": len(chunks),
        "model": model
    }
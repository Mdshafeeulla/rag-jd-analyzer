# ⚡ RAG Job Description Analyzer

A command-line tool that uses **Retrieval-Augmented Generation (RAG)** to 
intelligently compare your resume against any job description — running 
100% locally with no API keys or costs.

---

## 🧠 How It Works (RAG Pipeline)
```
resume.txt  →  Chunker  →  Embedder  →  Vector Store
                                               │
job_desc.txt  →  Embedder  →  Similarity Search
                                               │
                                        Top-K Chunks
                                               │
                                        Prompt Builder
                                               │
                                      Ollama (Local LLM)
                                               │
                                        CLI Analysis Report
```

1. **Chunk** — Resume is split into overlapping text chunks
2. **Embed** — Each chunk is converted to a 384-dim vector using `all-MiniLM-L6-v2`
3. **Retrieve** — Cosine similarity finds the most JD-relevant resume sections
4. **Generate** — Local LLM (Mistral/LLaMA) analyzes the retrieved context

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Search | Custom cosine similarity (NumPy) |
| Local LLM | Ollama (Mistral 7B) |
| CLI Interface | Python argparse + Rich |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running

### 1. Clone the repo
```bash
git clone https://github.com/Mdshafeeulla/rag-jd-analyzer.git
cd rag-jd-analyzer
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull the LLM model
```bash
ollama pull mistral
```

---

## ▶️ Usage
```bash
# Start Ollama in a separate terminal
ollama serve

# Run the analyzer
python main.py --resume data/sample_resume.txt --jd data/sample_jd.txt

# Optional flags
python main.py --resume data/resume.txt --jd data/jd.txt --model llama3 --top_k 7
```

---

## 📊 Sample Output
```
📊 Analysis Report
─────────────────────────────────────────
MATCH SCORE: 78/100

MATCHED SKILLS: Python, pandas, SQL, Power BI, data pipelines

MISSING SKILLS: Tableau, statistical modeling, A/B testing

TOP STRENGTH: Strong hands-on Python + automation experience

KEY GAP: No mention of Tableau (required in JD)

RESUME IMPROVEMENT:
"Built end-to-end data pipeline using Python (pandas, numpy) 
reducing manual reporting time by 40%"

INTERVIEW TIP: Expect "Walk me through a data pipeline you built."
Emphasize the automation and business impact.
```

---

## 📁 Project Structure
```
rag-jd-analyzer/
├── src/
│   ├── chunker.py        # Text chunking
│   ├── embedder.py       # Sentence embeddings
│   ├── vector_store.py   # Cosine similarity search
│   ├── llm.py            # Ollama integration
│   ├── prompt_builder.py # RAG prompt construction
│   └── pipeline.py       # Full pipeline orchestration
├── data/
│   ├── sample_resume.txt # Sample resume for testing
│   └── sample_jd.txt     # Sample job description
├── main.py               # CLI entry point
├── requirements.txt
└── README.md
```

---

## 💡 Key Concepts Implemented

- **RAG Architecture** — Retrieval before generation, not just prompting
- **Semantic Search** — Meaning-based retrieval using vector embeddings
- **Local LLM** — 100% private, no data leaves your machine
- **Modular Design** — Each component is independently testable

---

## 🔮 Roadmap

- [ ] PDF resume support
- [ ] Hybrid BM25 + semantic search
- [ ] Batch analyze multiple JDs at once
- [ ] Flask web interface

---

Built by [Md Shafeeulla](https://github.com/Mdshafeeulla) | 
Portfolio project demonstrating RAG implementation from scratch

# Phase 1 — Core Concepts Cheat Sheet

## The Pipeline at a Glance

```
Raw Text (PDF / URL / TXT)
        ↓
   [ Document Loader ]         ← LangChain loads & wraps text in Document objects
        ↓
   [ Text Splitter ]           ← Breaks docs into ~500 char chunks with 100 char overlap
        ↓
   [ Embedding Model ]         ← Converts each chunk → 384-dim float vector
   (all-MiniLM-L6-v2)
        ↓
   [ FAISS Index ]             ← Stores vectors + original text, enables fast search
        ↓
   [ Saved to Disk ]           ← index.faiss + index.pkl
```

---

## Key Concepts Explained

### 1. Document
```python
Document(
    page_content="The actual text goes here...",
    metadata={"source": "paper.pdf", "page": 3}
)
```
Every piece of text in LangChain is wrapped in a Document.
Metadata is key — it tells you *where* the answer came from.

---

### 2. Text Chunking
```
Original doc: 5,000 chars
                ↓
chunk 1: chars   0–500   (500 chars)
chunk 2: chars 400–900   (100 overlap)  ← shares 100 chars with chunk 1
chunk 3: chars 800–1300  (100 overlap)  ← shares 100 chars with chunk 2
```
**Why overlap?** If a key sentence falls at a chunk boundary,
overlap ensures it appears fully in at least one chunk.

**Rule of thumb:**
- chunk_size: 500–1000 for most use cases
- chunk_overlap: ~20% of chunk_size

---

### 3. Embeddings (the core idea)
```
"The cat sat on the mat"  →  [0.12, -0.34, 0.89, ..., 0.22]  # 384 numbers
"A feline rested on a rug" → [0.11, -0.31, 0.87, ..., 0.19]  # similar!
"Quantum physics paper"   →  [-0.45, 0.12, -0.67, ..., 0.03] # very different
```
Semantically similar text → numerically similar vectors.
Similarity is measured with **cosine similarity** (0 to 1, higher = more similar).

**Model comparison:**
| Model                   | Dims | Size  | Speed | Quality |
|------------------------|------|-------|-------|---------|
| all-MiniLM-L6-v2       | 384  | 80MB  | Fast  | Good    |
| all-mpnet-base-v2      | 768  | 420MB | Med   | Better  |
| BAAI/bge-small-en-v1.5 | 384  | 130MB | Fast  | Best small|
| BAAI/bge-large-en-v1.5 | 1024 | 1.3GB | Slow  | Best    |

---

### 4. FAISS Vector Store
```python
# Build (embeds all chunks — slow, do once)
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save (don't re-embed every restart)
vectorstore.save_local("vectorstore")

# Load (fast — just reads from disk)
vectorstore = FAISS.load_local("vectorstore", embeddings)

# Search (milliseconds — fast nearest-neighbor)
results = vectorstore.similarity_search("your query", k=3)
```

**Two search modes:**
- `similarity_search()` — top-k most similar chunks
- `max_marginal_relevance_search()` — top-k similar BUT diverse (avoids duplicate chunks)

---

### 5. L2 Distance Score (what FAISS returns)
```
score = 0.0   → identical vectors (exact match)
score = 0.5   → very similar
score = 1.0   → somewhat related
score = 2.0+  → unrelated
```
Lower = more similar (opposite of cosine similarity where higher = more similar)

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: faiss` | Not installed | `pip install faiss-cpu` |
| `allow_dangerous_deserialization` | LangChain security flag | Add `allow_dangerous_deserialization=True` to `load_local()` |
| Empty chunks | chunk_size too small | Increase chunk_size to ≥ 200 |
| Bad retrieval quality | Chunks too large | Decrease chunk_size to 300–500 |
| Slow embedding | Large model on CPU | Use MiniLM or add `device="cuda"` |

---

## What Carries into Phase 2

In Phase 2 (RAG Chain), you'll take this vectorstore and:
```python
# Convert vectorstore to a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# The retriever becomes the "R" in RAG — it fetches context for the LLM
```
Everything you built here becomes the knowledge base that powers the LLM's answers.

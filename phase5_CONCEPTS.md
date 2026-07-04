# Phase 5 — RAGAS + CrossEncoder Cheat Sheet

## The Two-Stage Retrieval Pipeline

```
Query
  │
  ▼
┌──────────────────────┐
│   FAISS (bi-encoder)  │  ← fast, embeds query + docs SEPARATELY
│   fetch_k = 20         │  Cast a WIDE net (recall-focused)
└──────────┬────────────┘
           │ 20 candidates
           ▼
┌──────────────────────┐
│  CrossEncoder rerank  │  ← slow but precise, scores (query, doc) TOGETHER
│   final_k = 4          │  Narrow to the BEST matches (precision-focused)
└──────────┬────────────┘
           │ 4 best docs
           ▼
        LLM (generation)
```

---

## Bi-Encoder vs Cross-Encoder — The Core Distinction

| | Bi-Encoder (FAISS/embeddings) | Cross-Encoder (reranker) |
|---|---|---|
| Input | Query and doc embedded SEPARATELY | (Query, doc) pair as ONE input |
| Speed | Fast — embed once, reuse forever | Slow — must run per query at search time |
| Accuracy | Good | Better — sees query+doc interaction |
| Scale | Works for millions of docs | Only practical for ~20-50 candidates |
| Use for | Initial candidate retrieval | Final re-ranking of a small set |

```python
# Bi-encoder: doc embeddings precomputed and cached
doc_vec = embed("FAISS uses L2 distance")        # computed once at ingestion
query_vec = embed("How does FAISS search?")       # computed at query time
score = cosine_similarity(query_vec, doc_vec)      # compare two separate vectors

# Cross-encoder: no precomputation possible — must see both together
score = cross_encoder.predict([["How does FAISS search?", "FAISS uses L2 distance"]])
# the model literally reads both texts as ONE input and outputs a relevance score
```

This is WHY you can't just use a cross-encoder for everything — you'd have
to run it against every document in your knowledge base for every query,
which doesn't scale past a few hundred documents.

---

## CrossEncoder Code Walkthrough

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = [
    ["How does FAISS search?", "FAISS computes L2 distance..."],
    ["How does FAISS search?", "The weather today is sunny..."],
]
scores = model.predict(pairs)
# scores = [4.2, -3.8]  -- relevant pair scores high, irrelevant scores low/negative
```

Scores from `ms-marco` models are NOT bounded [0,1] — they're raw logits.
Use them only for RELATIVE ranking (sort descending), not as probabilities.

---

## RAGAS Metrics — What Each One Actually Measures

```
                    ┌─────────────────────────────────────────┐
                    │              Your RAG System              │
                    │                                            │
   Question ───────►│  Retriever ──► Context ──► LLM ──► Answer │
                    │     │                          │            │
                    └─────┼──────────────────────────┼────────────┘
                          │                          │
              context_precision                faithfulness
              context_recall                    answer_relevancy
```

| Metric | Question it answers | Needs ground_truth? |
|--------|---------------------|----------------------|
| **faithfulness** | Is the answer grounded in retrieved context (no hallucination)? | No |
| **answer_relevancy** | Does the answer address what was actually asked? | No |
| **context_precision** | Are retrieved chunks actually relevant (no noise)? | No |
| **context_recall** | Did retrieval find ALL needed info? | **Yes** |

**Faithfulness** is the most important metric for catching hallucination —
it decomposes the answer into individual factual claims and checks each
one against the retrieved context.

---

## RAGAS Dataset Format — The Four Required Fields

```python
{
    "question":      ["What is RAG?"],
    "answer":        ["RAG combines retrieval with generation..."],
    "contexts":      [["RAG is a technique...", "It retrieves..."]],  # list of strings PER row
    "ground_truth":  ["RAG stands for Retrieval-Augmented Generation..."],
}
```

`contexts` is a list-of-lists — for each question, you provide the actual
chunks YOUR retriever returned (not pre-written reference contexts).
This is what makes RAGAS test your REAL system, not a hypothetical one.

---

## Reading the Diagnostic Output

```
⚠️  LOW faithfulness     → LLM hallucinating → tighten prompt + temperature=0
⚠️  LOW context_precision → noisy retrieval   → reduce k, add reranking
⚠️  LOW context_recall    → missing info       → increase k, smaller chunks
⚠️  LOW answer_relevancy  → off-topic answers  → tighten prompt focus
```

Each failure mode points to a DIFFERENT fix — this is why you measure
four separate metrics instead of one overall "quality" score. A system
can have perfect faithfulness (never hallucinates) while still scoring
terribly on context_recall (it's faithfully reporting that it doesn't
know things it actually should know, because retrieval missed them).

---

## Why the Out-of-Scope Test Question Matters

```python
{
    "question": "What is the capital of France?",
    "ground_truth": "I don't have enough information in the provided documents to answer this."
}
```

This question is intentionally NOT covered by your knowledge base. A good
RAG system should say "I don't know" rather than answering from the LLM's
own training data (which it definitely knows — "Paris"). If your system
answers "Paris" here, that's a faithfulness violation — proof your system
is leaking pretrained knowledge instead of staying grounded in retrieval.
This is one of the most common and dangerous RAG failure modes in production.

---

## Running the Comparison

```bash
python compare_reranking.py
```

Produces a table like:

```
Metric               Baseline   Reranked      Delta  Result
-------------------- ---------- ---------- ----------  ------
faithfulness              0.650      0.780     +0.130  ✅
answer_relevancy          0.720      0.750     +0.030  ✅
context_precision         0.550      0.810     +0.260  ✅
context_recall            0.680      0.690     +0.010  ➖
```

**Paste this directly into your portfolio README** — it's concrete,
numeric proof that you understand RAG evaluation, not just RAG construction.
Most fresher candidates can build a RAG pipeline; almost none can show you
quantified evidence that their improvements actually worked.

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ragas.exceptions.ExceptionInRunner` | Judge LLM rate-limited (Groq free tier) | Add delays, reduce test set size, or upgrade Groq tier |
| Scores all near 0 | Wrong dataset schema | Verify all 4 fields present, `contexts` is list-of-lists |
| CrossEncoder very slow | Large model + many candidates | Use `ms-marco-MiniLM-L-6-v2` (small) and cap fetch_k at 20-30 |
| `context_recall` always low | Ground truths too detailed | Write ground_truth using ONLY info actually in your documents |
| Reranking makes results worse | fetch_k too small | If fetch_k=4 already, reranking has nothing extra to find — increase fetch_k to 15-20 |

---

## What Carries Forward

This evaluation pipeline is exactly what you'd run in CI/CD for a real RAG
system — after every change to chunk_size, k, or prompt wording, rerun
`compare_reranking.py`-style scripts to confirm you didn't regress quality.
That's the difference between "I built a RAG system" and "I built and
validated a RAG system" — the second sentence is what gets you hired.

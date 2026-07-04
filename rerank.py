"""
Phase 5a: CrossEncoder Re-ranking
===================================
Adds a re-ranking step between retrieval and generation:

  Query → FAISS (fetch top-20, fast/imprecise) → CrossEncoder (rerank → top-4, slow/precise) → LLM

Why this matters:
  FAISS similarity search uses BI-ENCODERS — it embeds the query and each
  document SEPARATELY, then compares vectors. This is fast (one embedding
  per doc, computed once, reused forever) but loses information: the model
  never actually looks at the query and document TOGETHER.

  A CROSS-ENCODER takes (query, document) pairs as a SINGLE input and
  outputs a relevance score. It's far more accurate because it can model
  interactions between query and document terms — but it's slow, since you
  must run it once per (query, document) pair at query time.

  The standard solution: use the bi-encoder (FAISS) for fast CANDIDATE
  retrieval (cast a wide net, top-20), then use the cross-encoder to
  RE-RANK just those candidates down to the best top-4. Best of both worlds.
"""

from typing import List, Tuple
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder


# ─────────────────────────────────────────────
# STEP 1: LOAD THE CROSS-ENCODER MODEL
# ─────────────────────────────────────────────

def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """
    Load a CrossEncoder model trained specifically for relevance ranking.

    Model options (all free, run locally):
      "cross-encoder/ms-marco-MiniLM-L-6-v2"   → fast, good default (22M params)
      "cross-encoder/ms-marco-MiniLM-L-12-v2"  → slower, more accurate
      "BAAI/bge-reranker-base"                  → strong multilingual reranker
      "BAAI/bge-reranker-large"                 → best quality, slowest

    These models were trained on MS MARCO — millions of (query, relevant_passage,
    irrelevant_passage) triples — specifically to predict relevance scores.
    This is DIFFERENT from embedding models (which optimize for clustering
    similar texts together in vector space).
    """
    print(f"  🎯 Loading CrossEncoder: {model_name}")
    model = CrossEncoder(model_name, max_length=512)
    print(f"     → Model loaded (max_length=512 tokens per pair)")
    return model


# ─────────────────────────────────────────────
# STEP 2: RERANKING FUNCTION
# ─────────────────────────────────────────────

def rerank_documents(
    query: str,
    documents: List[Document],
    cross_encoder: CrossEncoder,
    top_k: int = 4,
) -> List[Tuple[Document, float]]:
    """
    Re-score and re-order a list of candidate documents using the CrossEncoder.

    Process:
      1. Build (query, doc_text) pairs for every candidate
      2. CrossEncoder.predict() scores ALL pairs in one batched call
      3. Sort by score descending
      4. Return the top_k highest-scoring documents

    Returns: List of (Document, score) tuples, sorted best-first.
    Scores are NOT bounded to [0,1] for ms-marco models — treat them as
    relative rankings, not probabilities (unless using a model with
    sigmoid-normalized output).
    """
    if not documents:
        return []

    # Build (query, passage) pairs — this is the CrossEncoder's required input format
    pairs = [[query, doc.page_content] for doc in documents]

    # Score all pairs at once (batched — much faster than looping)
    scores = cross_encoder.predict(pairs)

    # Pair each doc with its score, then sort descending
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return scored_docs[:top_k]


# ─────────────────────────────────────────────
# STEP 3: TWO-STAGE RETRIEVAL PIPELINE
# ─────────────────────────────────────────────

class RerankedRetriever:
    """
    Wraps a FAISS vectorstore + CrossEncoder into a single two-stage retriever.

    Stage 1 (recall):    FAISS fetches a WIDE candidate set (e.g. top-20)
                          Fast bi-encoder search, optimized for speed
    Stage 2 (precision): CrossEncoder re-scores and narrows to the BEST set
                          (e.g. top-4) — slower but far more accurate

    This is the industry-standard RAG retrieval pattern used in production
    systems (Cohere Rerank, Elasticsearch's "rerank" pipeline stage, etc.)
    """

    def __init__(
        self,
        vectorstore: FAISS,
        cross_encoder: CrossEncoder = None,
        fetch_k: int = 20,
        final_k: int = 4,
    ):
        self.vectorstore = vectorstore
        self.cross_encoder = cross_encoder or get_cross_encoder()
        self.fetch_k = fetch_k   # candidates to fetch in stage 1
        self.final_k = final_k   # final docs to return after stage 2

    def get_relevant_documents(self, query: str, verbose: bool = False) -> List[Document]:
        """
        Run the full two-stage pipeline. This method name matches LangChain's
        retriever interface so it can be used as a drop-in replacement.
        """
        # ── Stage 1: Fast candidate retrieval (bi-encoder / FAISS) ──
        candidates = self.vectorstore.similarity_search(query, k=self.fetch_k)

        if verbose:
            print(f"\n  📥 Stage 1 (FAISS): retrieved {len(candidates)} candidates")

        # ── Stage 2: Precise re-ranking (cross-encoder) ──
        reranked = rerank_documents(query, candidates, self.cross_encoder, top_k=self.final_k)

        if verbose:
            print(f"  🎯 Stage 2 (CrossEncoder): reranked down to top {len(reranked)}")
            for i, (doc, score) in enumerate(reranked, 1):
                src = doc.metadata.get("source", "?")
                print(f"     [{i}] score={score:.4f} | source={src} | {doc.page_content[:80].strip()}...")

        return [doc for doc, score in reranked]

    def invoke(self, query: str) -> List[Document]:
        """LCEL-compatible alias — lets you use this directly in a chain with | """
        return self.get_relevant_documents(query)


# ─────────────────────────────────────────────
# STEP 4: COMPARISON — BEFORE vs AFTER RERANKING
# ─────────────────────────────────────────────

def compare_retrieval(
    query: str,
    vectorstore: FAISS,
    cross_encoder: CrossEncoder,
    k: int = 4,
) -> None:
    """
    Side-by-side comparison: plain FAISS top-k vs reranked top-k.
    This is the single most convincing demo for why reranking matters —
    run it on your README / portfolio video.
    """
    print(f"\n{'='*60}")
    print(f"  Query: '{query}'")
    print(f"{'='*60}")

    # Plain FAISS (bi-encoder only)
    print(f"\n  📊 WITHOUT reranking (FAISS similarity_search top-{k}):")
    plain_results = vectorstore.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(plain_results, 1):
        src = doc.metadata.get("source", "?")
        print(f"     [{i}] L2_dist={score:.4f} | {src} | {doc.page_content[:70].strip()}...")

    # FAISS (wide net) + CrossEncoder rerank
    print(f"\n  🎯 WITH reranking (fetch 20, CrossEncoder rerank to top-{k}):")
    candidates = vectorstore.similarity_search(query, k=20)
    reranked = rerank_documents(query, candidates, cross_encoder, top_k=k)
    for i, (doc, score) in enumerate(reranked, 1):
        src = doc.metadata.get("source", "?")
        print(f"     [{i}] CE_score={score:.4f} | {src} | {doc.page_content[:70].strip()}...")

    print()
    print("  💡 Note: CrossEncoder often reorders results, surfacing docs that")
    print("     FAISS ranked lower but are actually MORE relevant to the query")
    print("     (because it reads query+doc together instead of comparing vectors).")


if __name__ == "__main__":
    # Quick standalone test — requires vectorstore from Phase 1
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    cross_encoder = get_cross_encoder()

    compare_retrieval("How does FAISS perform similarity search?", vectorstore, cross_encoder)
    compare_retrieval("What is the role of overlap in text chunking?", vectorstore, cross_encoder)

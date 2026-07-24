"""
Production Hybrid Retriever Module
====================================
Combines:
  1. Dense Vector Search (Pinecone / ChromaDB / FAISS)
  2. Sparse Keyword Search (BM25)
  3. Reciprocal Rank Fusion (RRF)
  4. Cross-Encoder Re-ranking (ms-marco-MiniLM-L-6-v2)

Features:
  - Parallel execution of Dense & BM25 retrievals via ThreadPoolExecutor.
  - RRF scoring formula: RRF_score(d) = sum(1 / (k + rank(d))) with k=60.
  - Returns detailed diagnostic metadata (Dense rank, BM25 rank, RRF score, Cross-Encoder score).
"""

import os
import pickle
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

_cross_encoder_singleton = None


def get_global_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """Load CrossEncoder model once as a process-wide singleton."""
    global _cross_encoder_singleton
    if _cross_encoder_singleton is None:
        print(f"🎯 Loading Cross-Encoder reranker: {model_name}...")
        _cross_encoder_singleton = CrossEncoder(model_name, max_length=512)
    return _cross_encoder_singleton


def reciprocal_rank_fusion(
    dense_results: List[Document],
    bm25_results: List[Document],
    rrf_k: int = 60,
    top_n: int = 40,
) -> List[Tuple[Document, Dict[str, Any]]]:
    """
    Combines dense and sparse search rankings using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(doc) = 1/(k + dense_rank) + 1/(k + bm25_rank)

    Returns:
        List of (Document, metadata_dict) sorted by RRF score descending.
    """
    scores: Dict[str, Dict[str, Any]] = {}

    def get_doc_id(doc: Document) -> str:
        src = str(doc.metadata.get("source", "doc")).strip()
        chunk_id = str(doc.metadata.get("chunk_id", "0")).strip()
        snippet_key = "".join(doc.page_content[:100].split()).lower()
        return f"{src}::chunk_{chunk_id}::{hash(snippet_key)}"

    # Score dense results
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = get_doc_id(doc)
        if doc_id not in scores:
            scores[doc_id] = {
                "doc": doc,
                "dense_rank": rank,
                "bm25_rank": None,
                "rrf_score": 0.0,
            }
        scores[doc_id]["dense_rank"] = rank
        scores[doc_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    # Score BM25 results
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = get_doc_id(doc)
        if doc_id not in scores:
            scores[doc_id] = {
                "doc": doc,
                "dense_rank": None,
                "bm25_rank": rank,
                "rrf_score": 0.0,
            }
        scores[doc_id]["bm25_rank"] = rank
        scores[doc_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    # Sort candidates by RRF score descending
    sorted_candidates = sorted(
        scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    result = []
    for item in sorted_candidates[:top_n]:
        meta = {
            "dense_rank": item["dense_rank"],
            "bm25_rank": item["bm25_rank"],
            "rrf_score": round(item["rrf_score"], 5),
        }
        result.append((item["doc"], meta))

    return result


class HybridRetriever:
    """
    Production Hybrid Search Retriever implementing parallel Dense + BM25 retrieval,
    RRF candidate fusion, and Cross-Encoder precision reranking.
    """

    def __init__(
        self,
        dense_retriever: Any,
        bm25_retriever: Optional[Any] = None,
        fetch_k: int = 40,
        final_k: int = 6,
        cross_encoder: Optional[CrossEncoder] = None,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.fetch_k = fetch_k
        self.final_k = final_k
        self.cross_encoder = cross_encoder or get_global_cross_encoder()

    def get_relevant_documents_with_scores(
        self, query: str
    ) -> List[Tuple[Document, Dict[str, Any]]]:
        """
        Runs full hybrid retrieval pipeline and returns documents with detailed score diagnostics.

        Returns:
            List[Tuple[Document, Dict[str, Any]]] where dict contains:
              - dense_rank
              - bm25_rank
              - rrf_score
              - cross_encoder_score
              - final_rank
        """
        dense_docs = []
        bm25_docs = []

        # Step 1: Parallel retrieval for Dense and BM25
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(self._fetch_dense, query)
            future_bm25 = executor.submit(self._fetch_bm25, query)

            dense_docs = future_dense.result()
            bm25_docs = future_bm25.result()

        # Fallback if BM25 is empty/unavailable
        if not bm25_docs and dense_docs:
            rrf_candidates = [
                (doc, {"dense_rank": i + 1, "bm25_rank": None, "rrf_score": round(1.0 / (60 + i + 1), 5)})
                for i, doc in enumerate(dense_docs[:self.fetch_k])
            ]
        elif not dense_docs and bm25_docs:
            rrf_candidates = [
                (doc, {"dense_rank": None, "bm25_rank": i + 1, "rrf_score": round(1.0 / (60 + i + 1), 5)})
                for i, doc in enumerate(bm25_docs[:self.fetch_k])
            ]
        elif not dense_docs and not bm25_docs:
            return []
        else:
            # Step 2: Reciprocal Rank Fusion (RRF)
            rrf_candidates = reciprocal_rank_fusion(
                dense_results=dense_docs,
                bm25_results=bm25_docs,
                rrf_k=60,
                top_n=self.fetch_k,
            )

        # Step 3: Cross-Encoder Reranking
        candidate_docs = [doc for doc, _ in rrf_candidates]
        pairs = [[query, doc.page_content] for doc in candidate_docs]

        ce_scores = self.cross_encoder.predict(pairs)

        # Combine RRF metadata with Cross-Encoder scores
        scored_results = []
        for i, (doc, meta) in enumerate(rrf_candidates):
            ce_score = float(ce_scores[i])
            meta["cross_encoder_score"] = round(ce_score, 4)
            scored_results.append((doc, meta))

        # Sort by Cross-Encoder score descending
        scored_results.sort(key=lambda x: x[1]["cross_encoder_score"], reverse=True)

        # Assign final top_k ranks and attach citation metadata
        final_results = []
        for rank, (doc, meta) in enumerate(scored_results[:self.final_k], start=1):
            meta["final_rank"] = rank
            doc.metadata["citation_id"] = rank
            final_results.append((doc, meta))

        return final_results

    def get_relevant_documents(self, query: str) -> List[Document]:
        """LangChain standard retriever method."""
        scored = self.get_relevant_documents_with_scores(query)
        return [doc for doc, _ in scored]

    def invoke(self, query: str) -> List[Document]:
        """LCEL invocation alias."""
        return self.get_relevant_documents(query)

    def _fetch_dense(self, query: str) -> List[Document]:
        try:
            if hasattr(self.dense_retriever, "invoke"):
                return self.dense_retriever.invoke(query)
            elif hasattr(self.dense_retriever, "get_relevant_documents"):
                return self.dense_retriever.get_relevant_documents(query)
            elif hasattr(self.dense_retriever, "similarity_search"):
                return self.dense_retriever.similarity_search(query, k=self.fetch_k)
            return []
        except Exception as e:
            print(f"⚠️ Dense retrieval failed: {e}")
            return []

    def _fetch_bm25(self, query: str) -> List[Document]:
        if self.bm25_retriever is None:
            return []
        try:
            if hasattr(self.bm25_retriever, "invoke"):
                return self.bm25_retriever.invoke(query)
            elif hasattr(self.bm25_retriever, "get_relevant_documents"):
                return self.bm25_retriever.get_relevant_documents(query)
            return []
        except Exception as e:
            print(f"⚠️ BM25 retrieval failed: {e}")
            return []


def test_hybrid_search():
    """Quick unit test function for validation."""
    print("Testing Hybrid Search...")
    sample_docs = [
        Document(page_content="Python is an interpreted programming language.", metadata={"source": "py.txt"}),
        Document(page_content="Retrieval-Augmented Generation combines search and LLMs.", metadata={"source": "rag.txt"}),
        Document(page_content="BM25 is a term-frequency keyword matching algorithm.", metadata={"source": "bm25.txt"}),
    ]
    from langchain_community.retrievers import BM25Retriever
    bm25 = BM25Retriever.from_documents(sample_docs)

    class MockDense:
        def invoke(self, q): return sample_docs

    retriever = HybridRetriever(dense_retriever=MockDense(), bm25_retriever=bm25, fetch_k=3, final_k=2)
    res = retriever.get_relevant_documents_with_scores("What is RAG?")
    print(f"Retrieved {len(res)} docs successfully.")
    for doc, meta in res:
        print(f" - [{doc.metadata.get('source')}] Score: {meta['cross_encoder_score']} | RRF: {meta['rrf_score']}")


if __name__ == "__main__":
    test_hybrid_search()

"""
Production Dense Retriever Module
====================================
Migrated to Vercel (No local ML).
Uses Pinecone Dense Vector Search.

Features:
  - Returns detailed diagnostic metadata (Dense rank).
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document

class HybridRetriever:
    """
    Production Search Retriever using purely Dense Search for Vercel Serverless.
    (Mocked as 'Hybrid' to preserve downstream compatibility).
    """

    def __init__(
        self,
        dense_retriever: Any,
        fetch_k: int = 40,
        final_k: int = 6,
    ):
        self.dense_retriever = dense_retriever
        self.fetch_k = fetch_k
        self.final_k = final_k

    def get_relevant_documents_with_scores(
        self, query: str
    ) -> List[Tuple[Document, Dict[str, Any]]]:
        """
        Runs dense retrieval and formats scores.
        """
        dense_docs = self._fetch_dense(query)

        scored_results = []
        for i, doc in enumerate(dense_docs[:self.final_k]):
            meta = {
                "dense_rank": i + 1,
                "bm25_rank": None,
                "rrf_score": 1.0 / (i + 1),
                "cross_encoder_score": 1.0,
                "final_rank": i + 1
            }
            doc.metadata["citation_id"] = i + 1
            scored_results.append((doc, meta))

        return scored_results

    def get_relevant_documents(self, query: str) -> List[Document]:
        scored = self.get_relevant_documents_with_scores(query)
        return [doc for doc, _ in scored]

    def invoke(self, query: str) -> List[Document]:
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

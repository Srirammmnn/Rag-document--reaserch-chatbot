"""
Phase 5c: Reranking Impact Comparison
========================================
Runs the SAME RAGAS evaluation twice — once with plain FAISS retrieval,
once with FAISS + CrossEncoder reranking — and prints a side-by-side
score comparison.

This is the single most valuable artifact for your portfolio README:
concrete proof that reranking measurably improved your RAG system,
backed by numbers instead of "it feels better."
"""

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

from rerank import RerankedRetriever, get_cross_encoder
from ragas_eval import build_test_set, generate_ragas_dataset, run_ragas_evaluation

from dotenv import load_dotenv
load_dotenv()


def build_answer_chain(llm, retriever):
    """
    Same RAG chain shape as Phase 2 — works with any retriever object.

    Handles two retriever types:
      - LangChain's native retriever (supports `|` piping directly)
      - Our custom RerankedRetriever (does NOT support `|`, so we wrap
        its .invoke() in a RunnableLambda instead)
    """
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using ONLY this context:\n{context}\nIf insufficient, say so."),
        ("human", "{question}"),
    ])

    # RerankedRetriever has .invoke() but isn't a true LangChain Runnable,
    # so always wrap the retrieval+format step in a RunnableLambda — this
    # works correctly for BOTH retriever types.
    retrieve_and_format = RunnableLambda(lambda q: format_docs(retriever.invoke(q)))

    chain = (
        RunnableParallel(
            context=retrieve_and_format,
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    print("=" * 70)
    print("  Reranking Impact: Plain FAISS vs FAISS + CrossEncoder")
    print("=" * 70)

    # ── Setup shared resources ──
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    test_set = build_test_set()

    # ── RUN 1: Plain FAISS retriever (baseline) ──
    print("\n" + "─" * 70)
    print("  RUN 1: Baseline — plain FAISS similarity_search (k=4)")
    print("─" * 70)

    plain_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    plain_chain = build_answer_chain(llm, plain_retriever)
    plain_dataset = generate_ragas_dataset(test_set, plain_chain, plain_retriever)
    plain_scores = run_ragas_evaluation(plain_dataset)

    # ── RUN 2: FAISS + CrossEncoder reranking ──
    print("\n" + "─" * 70)
    print("  RUN 2: FAISS (fetch_k=20) + CrossEncoder rerank (final_k=4)")
    print("─" * 70)

    cross_encoder = get_cross_encoder()
    reranked_retriever = RerankedRetriever(
        vectorstore, cross_encoder, fetch_k=20, final_k=4
    )
    reranked_chain = build_answer_chain(llm, reranked_retriever)
    reranked_dataset = generate_ragas_dataset(test_set, reranked_chain, reranked_retriever)
    reranked_scores = run_ragas_evaluation(reranked_dataset)

    # ── COMPARISON TABLE ──
    print("\n" + "=" * 70)
    print("  COMPARISON: Baseline vs Reranked")
    print("=" * 70)

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    comparison = pd.DataFrame({
        "metric": metric_cols,
        "baseline": [plain_scores[m].mean() for m in metric_cols],
        "reranked": [reranked_scores[m].mean() for m in metric_cols],
    })
    comparison["delta"] = comparison["reranked"] - comparison["baseline"]
    comparison["improved"] = comparison["delta"].apply(lambda x: "✅" if x > 0.01 else ("➖" if abs(x) <= 0.01 else "❌"))

    print()
    print(f"  {'Metric':<20} {'Baseline':>10} {'Reranked':>10} {'Delta':>10}  {'Result'}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}  {'-'*6}")
    for _, row in comparison.iterrows():
        print(f"  {row['metric']:<20} {row['baseline']:>10.3f} {row['reranked']:>10.3f} {row['delta']:>+10.3f}  {row['improved']}")

    avg_improvement = comparison["delta"].mean()
    print(f"\n  Average improvement across all metrics: {avg_improvement:+.3f}")

    # ── Save for README ──
    comparison.to_csv("reranking_comparison.csv", index=False)
    print(f"\n💾 Comparison table saved to reranking_comparison.csv")
    print("   (paste this table directly into your portfolio README)")

    print("\n" + "=" * 70)
    print("  ✅ Phase 5 reranking comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

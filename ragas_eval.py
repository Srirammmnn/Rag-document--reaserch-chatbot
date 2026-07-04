"""
Phase 5b: RAGAS Evaluation
=============================
Quantitatively evaluate your RAG pipeline instead of eyeballing answers.

RAGAS (Retrieval Augmented Generation Assessment) computes metrics by using
an LLM as a judge — it scores your system's outputs against a rubric for
each metric. This is the standard way RAG systems are evaluated in industry.

Metrics covered:
  - faithfulness         : Is the answer grounded in the retrieved context?
                            (catches hallucination — the #1 RAG failure mode)
  - answer_relevancy     : Does the answer actually address the question?
  - context_precision    : Are the retrieved chunks relevant (not noisy)?
  - context_recall       : Did retrieval find ALL the needed information?
                            (requires ground-truth reference answers)
"""

import os
from typing import List, Dict
from pathlib import Path

import pandas as pd
from datasets import Dataset

# RAGAS metrics
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# LangChain wrappers RAGAS needs to run its own LLM-judge calls
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────
# STEP 1: BUILD A TEST SET
# ─────────────────────────────────────────────

def build_test_set() -> List[Dict]:
    """
    A test set is a list of {question, ground_truth} pairs.

    ground_truth = the IDEAL answer, written by a human (you) who knows
    what's actually in the knowledge base. This is what makes evaluation
    OBJECTIVE rather than "does this look okay to me."

    In a real project: write 15-30 of these covering different difficulty
    levels (simple lookup, multi-hop reasoning, questions with no answer
    in the KB to test if the system correctly says "I don't know").
    """
    test_set = [
        {
            "question": "What is RAG and what problem does it solve?",
            "ground_truth": (
                "RAG (Retrieval-Augmented Generation) enhances LLMs by retrieving "
                "relevant context from an external knowledge base before generating "
                "a response. It solves the problem of LLMs having static training "
                "data by giving them access to up-to-date or private information "
                "without retraining."
            ),
        },
        {
            "question": "How does FAISS perform similarity search?",
            "ground_truth": (
                "FAISS performs similarity search by computing distance (typically L2 "
                "or cosine) between a query vector and stored document vectors, then "
                "returning the k vectors with the smallest distance as the most similar."
            ),
        },
        {
            "question": "Why is chunk overlap used when splitting documents?",
            "ground_truth": (
                "Chunk overlap is used to prevent losing context at chunk boundaries. "
                "By sharing some characters between consecutive chunks, key information "
                "that would otherwise be split across two chunks remains fully present "
                "in at least one of them."
            ),
        },
        {
            "question": "What are the two main phases of RAG?",
            "ground_truth": (
                "The two main phases of RAG are: (1) offline ingestion — chunking and "
                "embedding documents into a vector store, and (2) online retrieval — "
                "searching for relevant chunks at query time and passing them as "
                "context to the LLM."
            ),
        },
        {
            "question": "What is the capital of France?",
            # Intentionally OUT OF SCOPE — tests whether the system correctly
            # refuses to answer rather than hallucinating from the LLM's own knowledge
            "ground_truth": (
                "I don't have enough information in the provided documents to answer this."
            ),
        },
    ]
    return test_set


# ─────────────────────────────────────────────
# STEP 2: RUN YOUR RAG PIPELINE ON THE TEST SET
# ─────────────────────────────────────────────

def generate_ragas_dataset(
    test_set: List[Dict],
    answer_chain,           # your Phase 2 LCEL chain (or agent from Phase 3)
    retriever,               # your retriever (plain or RerankedRetriever from Phase 5a)
) -> Dataset:
    """
    For every test question:
      1. Run retrieval -> get the context chunks actually used
      2. Run the chain -> get the generated answer
      3. Package {question, answer, contexts, ground_truth} into RAGAS format

    RAGAS needs ALL FOUR fields to compute its metrics:
      question     -> what was asked
      answer       -> what your system generated
      contexts     -> what was retrieved (list of strings)
      ground_truth -> the ideal answer (for recall-based metrics)
    """
    print(f"\n🧪 Running RAG pipeline on {len(test_set)} test questions...")

    records = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for i, item in enumerate(test_set, 1):
        question = item["question"]
        print(f"  [{i}/{len(test_set)}] {question}")

        # Get retrieved contexts (handles both plain retriever and RerankedRetriever)
        if hasattr(retriever, "invoke"):
            retrieved_docs = retriever.invoke(question)
        else:
            retrieved_docs = retriever.get_relevant_documents(question)
        contexts = [doc.page_content for doc in retrieved_docs]

        # Get generated answer
        answer = answer_chain.invoke(question)

        records["question"].append(question)
        records["answer"].append(answer)
        records["contexts"].append(contexts)
        records["ground_truth"].append(item["ground_truth"])

    print("  ✅ Pipeline run complete")
    return Dataset.from_dict(records)


# ─────────────────────────────────────────────
# STEP 3: CONFIGURE RAGAS'S JUDGE LLM
# ─────────────────────────────────────────────

def get_ragas_models():
    """
    RAGAS metrics work by calling an LLM internally to JUDGE your outputs
    (e.g. "does this answer use information not present in the context?").
    You must wrap your LangChain LLM/embeddings so RAGAS can use them.

    Important: use a CAPABLE model for judging (llama-3.1-70b, not 8b) —
    weak judge models give unreliable scores.
    """
    judge_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    judge_embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(judge_embeddings)
    return ragas_llm, ragas_embeddings


# ─────────────────────────────────────────────
# STEP 4: RUN RAGAS EVALUATION
# ─────────────────────────────────────────────

def run_ragas_evaluation(dataset: Dataset) -> pd.DataFrame:
    """
    Compute all four RAGAS metrics on the dataset.

    What each metric actually measures:

    faithfulness (0-1, higher=better):
      Breaks the answer into individual claims, checks each claim against
      the retrieved context. Score = (claims supported by context) / (total claims).
      LOW score = hallucination — the LLM said things not in the context.

    answer_relevancy (0-1, higher=better):
      Generates several hypothetical questions FROM the answer, embeds them,
      compares to the original question's embedding. High similarity = the
      answer is actually on-topic and directly addresses the question
      (not padded with irrelevant tangents).

    context_precision (0-1, higher=better):
      Of the chunks retrieved, what fraction are actually relevant/useful?
      LOW score = your retriever is pulling in noise (bad k, bad chunk_size,
      or your reranking isn't working).

    context_recall (0-1, higher=better):
      Of the information needed to answer correctly (per ground_truth),
      what fraction was actually present in the retrieved context?
      LOW score = your retriever is MISSING relevant chunks entirely
      (the document might not have been chunked finely enough, or k is too low).
    """
    judge_llm, judge_embeddings = get_ragas_models()

    print("\n📊 Running RAGAS evaluation (this calls the judge LLM multiple times)...")

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    df = result.to_pandas()
    return df


# ─────────────────────────────────────────────
# STEP 5: REPORT GENERATION
# ─────────────────────────────────────────────

def print_evaluation_report(df: pd.DataFrame) -> None:
    """
    Print a readable summary — this is what goes in your README/portfolio.
    """
    print("\n" + "=" * 70)
    print("  RAGAS EVALUATION REPORT")
    print("=" * 70)

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    available_cols = [c for c in metric_cols if c in df.columns]

    print("\n📈 Average Scores:")
    for col in available_cols:
        avg = df[col].mean()
        bar = "█" * int(avg * 30)
        print(f"  {col:20s} {avg:.3f}  {bar}")

    print("\n📋 Per-Question Breakdown:")
    for idx, row in df.iterrows():
        print(f"\n  Q{idx+1}: {row['question'][:60]}...")
        for col in available_cols:
            flag = "⚠️ " if row[col] < 0.6 else "✅ "
            print(f"     {flag}{col:20s}: {row[col]:.3f}")

    print("\n" + "=" * 70)

    # Flag specific failure patterns
    print("\n🔍 Diagnostic Summary:")
    if "faithfulness" in df.columns and df["faithfulness"].mean() < 0.7:
        print("  ⚠️  LOW faithfulness -> LLM may be hallucinating. Tighten the system")
        print("      prompt: 'Use ONLY the provided context' + lower temperature.")
    if "context_precision" in df.columns and df["context_precision"].mean() < 0.7:
        print("  ⚠️  LOW context_precision -> retriever returning noisy/irrelevant chunks.")
        print("      Try: reduce k, add CrossEncoder reranking (see rerank.py), or")
        print("      use smaller chunk_size for more focused chunks.")
    if "context_recall" in df.columns and df["context_recall"].mean() < 0.7:
        print("  ⚠️  LOW context_recall -> retriever MISSING relevant information.")
        print("      Try: increase k, increase fetch_k before reranking, or check")
        print("      if the source document was actually ingested.")
    if "answer_relevancy" in df.columns and df["answer_relevancy"].mean() < 0.7:
        print("  ⚠️  LOW answer_relevancy -> answers are off-topic or too verbose.")
        print("      Tighten the prompt to be more direct/focused.")

    if all(df[c].mean() >= 0.7 for c in available_cols):
        print("  ✅ All metrics above 0.7 — solid baseline RAG performance!")


# ─────────────────────────────────────────────
# MAIN: FULL EVALUATION PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Phase 5b: RAGAS Evaluation")
    print("=" * 60)

    # ── Load Phase 1+2 components ──
    print("\n📂 Loading vectorstore + building chains...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

    # Build the RAG chain (reuse logic from Phase 2)
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using ONLY this context:\n{context}\nIf insufficient, say so."),
        ("human", "{question}"),
    ])

    answer_chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # ── Build test set ──
    test_set = build_test_set()

    # ── Run pipeline + collect outputs ──
    dataset = generate_ragas_dataset(test_set, answer_chain, retriever)

    # ── Evaluate with RAGAS ──
    df = run_ragas_evaluation(dataset)

    # ── Report ──
    print_evaluation_report(df)

    # ── Save for README ──
    df.to_csv("ragas_results.csv", index=False)
    print(f"\n💾 Full results saved to ragas_results.csv")

    print("\n" + "=" * 60)
    print("  ✅ Phase 5b Complete!")
    print("  Next: Compare WITH vs WITHOUT reranking using rerank.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

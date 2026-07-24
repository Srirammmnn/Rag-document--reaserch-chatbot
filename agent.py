"""
Phase 3 & Production Agent Architecture: LangGraph Router + Hybrid Search + Verified Citations
=============================================================================================
Flow: User -> Router Node -> [RAG | LLM | Web | Math | Python] -> END

Optimizations & Features:
  - HybridRetriever singleton combining Dense Search (Pinecone/FAISS) + BM25 + RRF + CrossEncoder.
  - Strict grounded prompt requiring inline citation tags [1], [2].
  - Automated Citation Verification using citation_verifier.py.
  - Clean state management in LangGraph.
"""

import os
import sys
import re
import pickle
import concurrent.futures
from pathlib import Path
from typing import Annotated, List, Sequence, TypedDict, Dict, Any, Optional

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool

from hybrid_retriever import HybridRetriever, get_global_cross_encoder
from citation_verifier import verify_claims_against_sources
from ingest import get_embedding_model

from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    route: str
    citations: Optional[List[Dict[str, Any]]]
    groundedness_score: Optional[float]
    retrieval_diagnostics: Optional[List[Dict[str, Any]]]


# ─────────────────────────────────────────────
# HYBRID RETRIEVER SINGLETON
# ─────────────────────────────────────────────

_hybrid_retriever_singleton = None

def init_retriever() -> Optional[HybridRetriever]:
    """Initialize and cache the global HybridRetriever singleton."""
    global _hybrid_retriever_singleton

    print("🧩 Initializing Production Hybrid Retriever (Dense + BM25 + RRF + Reranker)...")
    vectorstore_dir = Path(__file__).parent / "vectorstore"
    embeddings = get_embedding_model()

    dense_retriever = None

    # Option 1: Pinecone Cloud
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME") or "rag"
    if pinecone_key and index_name:
        try:
            from pinecone import Pinecone as PineconeClient
            from langchain_pinecone import Pinecone
            pc = PineconeClient(api_key=pinecone_key)
            existing_indexes = [idx.name for idx in pc.list_indexes()]

            if index_name in existing_indexes:
                index_obj = pc.Index(index_name)
                vectorstore = Pinecone(index=index_obj, embedding=embeddings)
                dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 30})
                print(f"  ☁️ Dense Retriever: Connected to Pinecone index '{index_name}'")
            else:
                print(f"  ⚠️ Pinecone index '{index_name}' does not exist on cloud yet.")
        except Exception as e:
            print(f"  ⚠️ Pinecone connection failed ({e})")

    # Option 2: Local ChromaDB / FAISS Fallback
    if dense_retriever is None:
        chroma_dir = vectorstore_dir / "chroma_db"
        if chroma_dir.exists():
            try:
                from langchain_chroma import Chroma
                vectorstore = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)
                dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 30})
                print("  💾 Dense Retriever: Loaded local ChromaDB store")
            except Exception as e:
                print(f"  ⚠️ ChromaDB load failed: {e}")

    if dense_retriever is None:
        faiss_dir = vectorstore_dir / "faiss_index"
        if faiss_dir.exists():
            try:
                from langchain_community.vectorstores import FAISS
                vectorstore = FAISS.load_local(str(faiss_dir), embeddings, allow_dangerous_deserialization=True)
                dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 30})
                print("  💾 Dense Retriever: Loaded local FAISS store")
            except Exception as e:
                print(f"  ⚠️ FAISS load failed: {e}")

    # Load BM25 Sparse Index from chunks.pkl
    bm25_retriever = None
    chunks_path = vectorstore_dir / "chunks.pkl"
    if chunks_path.exists():
        try:
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            if chunks:
                bm25_retriever = BM25Retriever.from_documents(chunks)
                bm25_retriever.k = 30
                print(f"  📝 Sparse Retriever: Loaded BM25 index with {len(chunks)} chunks")
        except Exception as e:
            print(f"  ⚠️ BM25 load failed: {e}")

    if dense_retriever is None and bm25_retriever is None:
        print("  ⚠️ No retriever could be initialized. Please ingest documents first.")
        _hybrid_retriever_singleton = None
        return None

    _hybrid_retriever_singleton = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        fetch_k=40,
        final_k=6,
    )
    print("  ✅ HybridRetriever Ready!")
    return _hybrid_retriever_singleton


def get_hybrid_retriever() -> Optional[HybridRetriever]:
    global _hybrid_retriever_singleton
    if _hybrid_retriever_singleton is None:
        return init_retriever()
    return _hybrid_retriever_singleton


# ─────────────────────────────────────────────
# ROUTER NODE
# ─────────────────────────────────────────────

def llm_based_route(question: str) -> str:
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=15)
        system_prompt = (
            "You are an expert intent classifier for a smart assistant.\n"
            "Route the user's question to ONE of the following nodes:\n"
            "- 'rag': ONLY for questions specifically asking about the user's personal info, resume, skills, experience, or extracting information from user's uploaded documents.\n"
            "- 'web': For current events, recent news, live scores, or weather.\n"
            "- 'math': For pure mathematical calculations (e.g. 'calculate 5+5').\n"
            "- 'python': For writing or executing python scripts and code.\n"
            "- 'llm': For general knowledge questions (like capitals, history, AI definitions), general chit-chat, greetings, conversational responses, or subjective questions.\n\n"
            "Respond ONLY with the single word corresponding to the node: rag, web, math, python, or llm."
        )
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]).content.strip().lower()

        valid_routes = ["rag", "llm", "web", "math", "python"]
        for r in valid_routes:
            if r in response:
                return r
    except Exception as e:
        print(f"   -> LLM Routing failed ({e}), falling back to 'rag'")
        
    return "rag"

def router_node(state: AgentState):
    """Classifies incoming query intent to route to appropriate processing node using ONLY LLM."""
    messages = state["messages"]
    question = messages[-1].content

    print(f"🚦 Router Node: LLM Classifying intent for query: '{question}'...")
    destination = llm_based_route(question)
    print(f"   -> LLM Routed to: {destination}")

    synthetic_tool = AIMessage(
        content="",
        tool_calls=[{"name": f"routed_to_{destination}", "args": {"query": question}, "id": "route_1"}]
    )
    return {"messages": [synthetic_tool], "route": destination}


# ─────────────────────────────────────────────
# RAG WORKER NODE WITH CITATION SYNTHESIS
# ─────────────────────────────────────────────

def preprocess_retrieval_query(query: str) -> str:
    """Strips excessive whitespace for optimal vector & keyword search while preserving natural phrasing."""
    return re.sub(r'\s+', ' ', query).strip()


def rag_node(state: AgentState):
    print("📚 RAG Node: Hybrid Retrieval & Citation Generation...")

    question = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break
    if not question:
        question = state["messages"][-2].content

    # Query Expansion: Generate keywords to boost BM25 retrieval
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        expansion_prompt = (
            f"Analyze this search query: '{question}'. Generate a list of core keywords, synonyms, and variations. "
            f"If there are compounded words like 'myprojects', split them into 'my projects'. "
            f"Return ONLY the keywords and variations separated by spaces, with no conversational text."
        )
        expanded_keywords = llm.invoke([HumanMessage(content=expansion_prompt)]).content.strip()
        print(f"   -> Query Expansion: '{expanded_keywords}'")
        search_query = preprocess_retrieval_query(f"{question} {expanded_keywords}")
    except Exception as e:
        print(f"   -> Query expansion failed ({e}), using raw query.")
        search_query = preprocess_retrieval_query(question)
    retriever = get_hybrid_retriever()

    if retriever is None:
        err_msg = AIMessage(content="Knowledge base is empty. Please upload documents first.")
        return {"messages": [err_msg], "citations": [], "groundedness_score": 0.0, "retrieval_diagnostics": []}

    # Step 1: Hybrid Retrieval with detailed score diagnostics using full natural question
    results_with_scores = retriever.get_relevant_documents_with_scores(search_query)

    context_chunks = []
    diagnostics = []
    docs_for_verification = []

    for rank, (doc, meta) in enumerate(results_with_scores, start=1):
        doc.metadata["citation_id"] = rank
        docs_for_verification.append(doc)

        src = doc.metadata.get("source", "document")
        page = doc.metadata.get("page", 1)
        context_chunks.append(f"[{rank}] (Source: {src}, Page {page}):\n{doc.page_content.strip()}")

        diagnostics.append({
            "citation_id": rank,
            "source": src,
            "page": page,
            "dense_rank": meta.get("dense_rank"),
            "bm25_rank": meta.get("bm25_rank"),
            "rrf_score": meta.get("rrf_score"),
            "cross_encoder_score": meta.get("cross_encoder_score"),
            "snippet": doc.page_content[:150] + "...",
        })

    context_str = "\n\n".join(context_chunks)

    # Step 2: System prompt demanding grounded synthesis & inline citation tags [1], [2]
    system_prompt = (
        "You are an expert AI assistant answering the user's query.\n\n"
        "INSTRUCTIONS:\n"
        "1. Provide a direct, clear, and comprehensive answer using the Context Documents provided below.\n"
        "2. CRITICAL: Do NOT use conversational filler like 'Based on the provided documents', 'According to the context', or 'The documents state'. Present the information directly as if it were your own innate knowledge.\n"
        "3. Every factual claim drawn from the context MUST include inline numeric citations like [1], [2].\n"
        "4. If the context does not contain the answer, say 'I do not have enough information to answer this based on the provided documents.' Do NOT use your general knowledge.\n\n"
        f"Context Documents:\n{context_str}"
    )

    clean_history = [
        m for m in state["messages"][:-2]
        if isinstance(m, (HumanMessage, AIMessage)) and not getattr(m, "tool_calls", None)
    ][-4:]

    messages_to_send = [SystemMessage(content=system_prompt)] + clean_history + [HumanMessage(content=question)]

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=750)
    response = llm.invoke(messages_to_send)

    # Step 3: Citation Verification
    citations, groundedness_score = verify_claims_against_sources(response.content, docs_for_verification)
    print(f"   -> Groundedness Score: {groundedness_score*100:.1f}% ({len(citations)} citations verified)")

    tm = ToolMessage(content=f"Found {len(docs_for_verification)} grounded chunks.", tool_call_id="route_1", name="routed_to_rag")
    ai_msg = AIMessage(content=response.content)

    return {
        "messages": [tm, ai_msg],
        "citations": citations,
        "groundedness_score": groundedness_score,
        "retrieval_diagnostics": diagnostics,
    }


# ─────────────────────────────────────────────
# OTHER WORKER NODES
# ─────────────────────────────────────────────

def llm_node(state: AgentState):
    print("🧠 General LLM Node...")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, max_tokens=512)
    response = llm.invoke(state["messages"][:-1])
    tm = ToolMessage(content="LLM Answered.", tool_call_id="route_1", name="routed_to_llm")
    return {"messages": [tm, AIMessage(content=response.content)], "citations": [], "groundedness_score": 1.0}


def web_node(state: AgentState):
    print("🌐 Web Search Node...")
    question = state["messages"][-2].content if len(state["messages"]) >= 2 else state["messages"][-1].content
    search = DuckDuckGoSearchRun()
    search_result = search.invoke(question)

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
    prompt = f"Based on web search results:\n{search_result}\n\nAnswer question: {question}"
    response = llm.invoke([HumanMessage(content=prompt)])

    tm = ToolMessage(content="Searched Web.", tool_call_id="route_1", name="routed_to_web")
    return {"messages": [tm, AIMessage(content=response.content)], "citations": [], "groundedness_score": 1.0}


def math_node(state: AgentState):
    print("🧮 Math Node...")
    question = state["messages"][-2].content if len(state["messages"]) >= 2 else state["messages"][-1].content
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    expr = llm.invoke([HumanMessage(content=f"Extract pure math expression from: '{question}'. Return ONLY expression.")]).content.strip()

    try:
        allowed = set("0123456789+-*/(). ")
        ans = str(eval(expr, {"__builtins__": {}}, {})) if all(c in allowed for c in expr) else "Invalid expression"
    except Exception as e:
        ans = str(e)

    tm = ToolMessage(content="Calculated.", tool_call_id="route_1", name="routed_to_math")
    return {"messages": [tm, AIMessage(content=f"The result of `{expr}` is **{ans}**.")]}


def python_node(state: AgentState):
    print("🐍 Python Node...")
    question = state["messages"][-2].content if len(state["messages"]) >= 2 else state["messages"][-1].content
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    code = llm.invoke([HumanMessage(content=f"Write Python code to solve: {question}. Return ONLY executable Python code.")]).content.replace("```python", "").replace("```", "").strip()

    repl = PythonREPLTool()
    try:
        result = repl.invoke(code)
    except Exception as e:
        result = str(e)

    final_ans = f"```python\n{code}\n```\nOutput:\n```\n{result}\n```"
    tm = ToolMessage(content="Executed Python.", tool_call_id="route_1", name="routed_to_python")
    return {"messages": [tm, AIMessage(content=final_ans)]}


def route_decision(state: AgentState):
    return state["route"]


# ─────────────────────────────────────────────
# BUILD LANGGRAPH GRAPH
# ─────────────────────────────────────────────

def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("llm", llm_node)
    graph.add_node("web", web_node)
    graph.add_node("math", math_node)
    graph.add_node("python", python_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_decision,
        {"rag": "rag", "llm": "llm", "web": "web", "math": "math", "python": "python"}
    )
    graph.add_edge("rag", END)
    graph.add_edge("llm", END)
    graph.add_edge("web", END)
    graph.add_edge("math", END)
    graph.add_edge("python", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_agent_graph()
    print("✅ Production Agent Graph built successfully.")

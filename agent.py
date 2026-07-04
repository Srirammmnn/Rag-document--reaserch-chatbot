"""
Phase 3: LangGraph Router Architecture + Hybrid Search
======================================================
Flow: User → Router (rules + LLM fallback) → [RAG | LLM | Web | Math | Python] → END

Optimizations applied:
  - Two-stage router: keyword rules (0ms) → LLM fallback only when ambiguous
  - HybridRetriever is a global singleton (loaded once at startup, not per-request)
  - CrossEncoder pre-loaded globally
  - BM25 + Pinecone run in parallel via ThreadPoolExecutor
  - Generation capped at 512 tokens
"""

import os
import sys
import re
import pickle
from typing import Annotated, List, Sequence, TypedDict
from pathlib import Path

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from pydantic import BaseModel, Field

# LangGraph core
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

# LangChain core
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langchain_groq import ChatGroq
from langchain_pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

# Hybrid Search
from langchain_community.retrievers import BM25Retriever

# Tools
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import CrossEncoder
import concurrent.futures


# ─────────────────────────────────────────────
# CROSS ENCODER — GLOBAL SINGLETON
# ─────────────────────────────────────────────

_cross_encoder = None

def load_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        print("  🎯 Loading CrossEncoder globally (once)...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    return _cross_encoder


# ─────────────────────────────────────────────
# HYBRID RETRIEVER
# ─────────────────────────────────────────────

class HybridRetriever:
    def __init__(self, bm25_retriever, pinecone_retriever):
        self.bm25 = bm25_retriever
        self.pinecone = pinecone_retriever
        self.cross_encoder = load_cross_encoder()

    def invoke(self, query: str):
        # BM25 and Pinecone fetch in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(self.bm25.invoke, query)
            pinecone_future = executor.submit(self.pinecone.invoke, query)
            bm25_docs = bm25_future.result()
            pinecone_docs = pinecone_future.result()

        # Deduplicate
        unique_docs = {}
        for doc in bm25_docs + pinecone_docs:
            unique_docs[doc.page_content] = doc
        candidates = list(unique_docs.values())

        if not candidates:
            return []

        # CrossEncoder reranking
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        scored_docs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        # Return top 5 — enough for multi-doc coverage while keeping prompt tight
        return [doc for doc, _ in scored_docs[:5]]


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    route: str


# ─────────────────────────────────────────────
# HYBRID RETRIEVER — GLOBAL SINGLETON
# Built ONCE at FastAPI startup, reused every request.
# (Was the critical 15s bottleneck when built per-request)
# ─────────────────────────────────────────────

_hybrid_retriever = None

def init_retriever():
    """Pre-warm the hybrid retriever. Called once at FastAPI @startup."""
    global _hybrid_retriever
    if _hybrid_retriever is not None:
        return _hybrid_retriever

    print("🧩 Initializing Hybrid Search Singleton (Pinecone + BM25)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not pinecone_api_key or not index_name:
        print("⚠️ Pinecone credentials missing. Retriever not initialized.")
        return None

    vectorstore = Pinecone(index_name=index_name, embedding=embeddings)
    pinecone_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    chunks_path = Path(__file__).parent / "vectorstore" / "chunks.pkl"
    if chunks_path.exists():
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 8
        _hybrid_retriever = HybridRetriever(bm25_retriever, pinecone_retriever)
        print("  ✅ Hybrid retriever ready (Pinecone + BM25)")
    else:
        print("⚠️ chunks.pkl not found! Using Dense-only Pinecone search.")
        _hybrid_retriever = pinecone_retriever

    return _hybrid_retriever

def get_hybrid_retriever():
    """Return the cached singleton. Calls init_retriever() on first use."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        return init_retriever()
    return _hybrid_retriever


# ─────────────────────────────────────────────
# PRODUCTION ROUTER — THREE-TIER SYSTEM
# Tier 0: First-person absolute override (regex, instant)
# Tier 1: Domain keyword rules (string match, instant)
# Tier 2: LLM fallback (only for pure ambiguous queries, rag-biased)
# ─────────────────────────────────────────────

def router_node(state: AgentState):
    """
    Pure LLM-based router using Groq.
    Classifies the user request directly using the LLM to determine the correct tool/node.
    """
    messages = state["messages"]
    question = messages[-1].content

    print("🚦 Router (LLM Classification)...")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=10)
    system = (
        "You are a strict query router for an AI assistant.\n"
        "Analyze the user's query and classify it into EXACTLY ONE of the following categories:\n"
        "1. rag : STRONGLY DEFAULT TO THIS. Use this if the query contains 'I', 'me', 'my' (e.g., 'who am I', 'what is my GPA'), asks about personal details, resume, background, education, or facts from uploaded documents.\n"
        "2. math : If the query involves evaluating purely mathematical arithmetic expressions.\n"
        "3. python : If the query asks to write or execute Python code.\n"
        "4. web : If the query requires live/current internet data, news, or real-time search.\n"
        "5. llm : Only use this for general world knowledge, basic greetings ('hello'), or completely abstract topics that have absolutely nothing to do with the user's personal data.\n\n"
        "Output ONLY the category name (rag, math, python, web, or llm) in lowercase, with no other text."
    )
    prompt = f"{system}\nQuery: {question}\nRoute:"
    resp = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    
    # Extract the first word to ensure clean routing
    first = resp.split()[0] if resp.split() else "llm"
    valid = {"rag", "llm", "web", "math", "python"}
    destination = first if first in valid else "llm"
    
    print(f"   → LLM classified route as → {destination}")

    synthetic_tool_call = AIMessage(
        content="",
        tool_calls=[{"name": f"routed_to_{destination}", "args": {"query": question}, "id": "route_1"}]
    )
    return {"messages": [synthetic_tool_call], "route": destination}# ─────────────────────────────────────────────
# WORKER NODES
# ─────────────────────────────────────────────

def query_transform(question: str) -> str:
    """HyDE + rewrite (kept for reference, bypassed in rag_node for speed)."""
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
    prompt = (
        "You are an expert AI searching a vector database.\n"
        "1. Rewrite the query with relevant keywords.\n"
        "2. Write a short hypothetical answer containing expected terminology.\n\n"
        f"User Query: {question}\n\n"
        "Return ONLY the combined rewritten query + hypothetical answer."
    )
    return llm.invoke([HumanMessage(content=prompt)]).content

def hallucination_gate(context: str, answer: str, question: str) -> str:
    """RAGAS-style faithfulness check (kept for reference, bypassed for speed)."""
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
    prompt = (
        "You are an impartial judge evaluating faithfulness.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAI Answer: {answer}\n\n"
        "Rules:\n"
        "1. If the answer says it doesn't know → PASS\n"
        "2. If the answer is supported by the context → PASS\n"
        "3. If the answer has unsupported claims → FAIL\n\n"
        "Reply with EXACTLY one word: PASS or FAIL."
    )
    evaluation = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    if "PASS" in evaluation:
        return answer
    print(f"⚠️ Hallucination gate blocked response. Judge: {evaluation}")
    return "I could not generate a verified answer from the provided documents."


def preprocess_retrieval_query(query: str) -> str:
    """Strip question templates and stopwords to optimize retrieval recall."""
    q = query.lower().strip()
    prefixes = [
        r'\bwhat\s+is\s+(my|the|a|an)\b',
        r'\bwhat\s+is\b',
        r'\btell\s+me\s+(about|my|about\s+my|the)\b',
        r'\btell\s+me\b',
        r'\bcould\s+you\s+check\s+my\b',
        r'\bcould\s+you\s+(tell|check|show)\b',
        r'\bcheck\s+my\b',
        r'\bdo\s+i\s+have\s+(a|an|any)?\b',
        r'\bdo\s+i\s+have\b',
        r'\bshow\s+(my|me|the)\b',
        r'\bgive\s+me\s+(my|the)\b',
        r'\bfind\s+(my|the)\b',
        r'\bplease\s+(check|tell|show|give|find)\b',
        r'\bplease\b',
    ]
    for p in prefixes:
        q = re.sub(p, '', q)
    # Strip common small stop words that dilute search
    q = re.sub(r'\b(my|of|about|for|the|a|an|is|are|am|i|do|have|check|tell|show|give|find)\b', ' ', q)
    q = re.sub(r'\s+', ' ', q).strip()
    q = q.replace('?', '').strip()
    return q if q else query


def rag_node(state: AgentState):
    print("📚 RAG Node (Hybrid Search, speed-optimized)...")
    
    # Bug fix: scan backwards to find the actual last HumanMessage
    # (messages[-2] breaks with multi-turn session history)
    question = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break
    if not question:
        question = state["messages"][-2].content  # fallback

    # Retrieval — singleton retriever, parallel BM25+Pinecone, CrossEncoder rerank
    retriever = get_hybrid_retriever()
    clean_query = preprocess_retrieval_query(question)
    print(f"   🔍 Retrieval query: {question!r} -> refined: {clean_query!r}")
    docs = retriever.invoke(clean_query)

    context_parts = [
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    ]
    context_str = "\n\n".join(context_parts)

    # Tight prompt, trimmed history, capped tokens
    system_prompt = (
        "You are a professional assistant answering questions about personal documents.\n"
        "Use the provided context AND the conversation history to answer the user's latest question.\n"
        "If the context does not contain the exact answer, say so clearly, but proactively mention any closely related details that ARE present in the context.\n\n"
        f"Context:\n{context_str}"
    )
    history = list(state["messages"][:-2])[-4:]
    messages_to_send = [SystemMessage(content=system_prompt)] + history + [HumanMessage(content=question)]

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
    response = llm.invoke(messages_to_send)

    tm = ToolMessage(content=f"Found {len(docs)} documents.", tool_call_id="route_1", name="routed_to_rag")
    return {"messages": [tm, AIMessage(content=response.content)]}


def llm_node(state: AgentState):
    print("🧠 General LLM Node...")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, max_tokens=512)
    response = llm.invoke(state["messages"][:-1])
    tm = ToolMessage(content="LLM Answered.", tool_call_id="route_1", name="routed_to_llm")
    return {"messages": [tm, AIMessage(content=response.content)]}


def web_node(state: AgentState):
    print("🌐 Web Search Node...")
    question = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break
    if not question:
        question = state["messages"][-2].content

    search = DuckDuckGoSearchRun()
    search_result = search.invoke(question)

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
    prompt = f"Based on this web search result:\n{search_result}\n\nAnswer: {question}"
    response = llm.invoke([HumanMessage(content=prompt)])

    tm = ToolMessage(content="Searched Web.", tool_call_id="route_1", name="routed_to_web")
    return {"messages": [tm, AIMessage(content=response.content)]}


def math_node(state: AgentState):
    print("🧮 Math Node...")
    question = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break
    if not question:
        question = state["messages"][-2].content

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    extract_prompt = f"Extract only the mathematical expression to evaluate from: '{question}'. Return ONLY the expression."
    expr = llm.invoke([HumanMessage(content=extract_prompt)]).content.strip()

    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expr):
            ans = "Error: invalid math expression."
        else:
            ans = str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        ans = str(e)

    tm = ToolMessage(content="Calculated.", tool_call_id="route_1", name="routed_to_math")
    return {"messages": [tm, AIMessage(content=f"The result of `{expr}` is **{ans}**.")]}


def python_node(state: AgentState):
    print("🐍 Python Node...")
    question = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            question = msg.content
            break
    if not question:
        question = state["messages"][-2].content

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    code_prompt = f"Write Python code to solve: {question}. Return ONLY valid Python, no markdown."
    code = llm.invoke([HumanMessage(content=code_prompt)]).content.replace("```python", "").replace("```", "").strip()

    repl = PythonREPLTool()
    try:
        result = repl.invoke(code)
    except Exception as e:
        result = str(e)

    final_ans = f"```python\n{code}\n```\nOutput:\n```\n{result}\n```"
    tm = ToolMessage(content="Ran Python.", tool_call_id="route_1", name="routed_to_python")
    return {"messages": [tm, AIMessage(content=final_ans)]}


# ─────────────────────────────────────────────
# ROUTING LOGIC
# ─────────────────────────────────────────────

def route_decision(state: AgentState):
    return state["route"]


# ─────────────────────────────────────────────
# BUILD GRAPH
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
    print("\n✅ Router Graph built successfully!")

    question = "What are my certifications?"
    print(f"\nUser: {question}")
    res = app.invoke({"messages": [HumanMessage(content=question)]})
    print(f"Final Answer: {res['messages'][-1].content}")

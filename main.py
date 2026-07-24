"""
Phase 4 & Production FastAPI Backend Server
===========================================
Exposes Production RAG System APIs:
  - POST /ingest             -> Upload document, run semantic chunking & hybrid indexing
  - POST /chat               -> Blocking RAG Q&A with grounded answers & citations
  - POST /chat/stream        -> Real-time SSE streaming of thinking, answer tokens, and verified citations
  - GET  /retrieval/inspect  -> Side-by-side diagnostic inspection of Dense vs BM25 vs RRF vs Reranker
  - POST /verify-citation    -> Standalone citation & claim verification
  - GET  /sources            -> List active documents & chunk statistics
  - GET  /health             -> Backend health & vectorstore status check
"""

import os
import sys
import json
import pickle
import shutil
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from ingest import run_ingestion_pipeline, delete_document_from_stores, get_embedding_model
from agent import build_agent_graph, init_retriever, get_hybrid_retriever, _hybrid_retriever_singleton
import agent as agent_module
from citation_verifier import verify_claims_against_sources

from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────
# FASTAPI APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="NeuRAG Production Engine API",
    description="Production-grade RAG System with Hybrid Search (Dense + BM25 + RRF + Cross-Encoder) & Citation Verification",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTORSTORE_PATH = Path(__file__).parent / "vectorstore"
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_agent_graph = None

def get_agent_graph():
    global _agent_graph
    if _agent_graph is None:
        print("🏗️ Building LangGraph agent graph...")
        _agent_graph = build_agent_graph()
    return _agent_graph


# ─────────────────────────────────────────────
# REDIS / IN-MEMORY CACHE & SESSION HISTORY
# ─────────────────────────────────────────────
import redis
REDIS_CLIENT = None
try:
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    temp_client = redis.from_url(redis_url, socket_timeout=1)
    temp_client.ping()
    REDIS_CLIENT = temp_client
    print("🟢 Redis connected for enterprise session storage.")
except Exception:
    print("🟡 Redis not available. Utilizing in-memory session store.")
    REDIS_CLIENT = None

SESSIONS: Dict[str, list] = {}
SEMANTIC_CACHE: list = []

def get_session_history(session_id: str) -> list:
    if REDIS_CLIENT:
        data = REDIS_CLIENT.get(f"session:{session_id}")
        return pickle.loads(data) if data else []
    return SESSIONS.get(session_id, [])

def save_session_history(session_id: str, messages: list):
    if REDIS_CLIENT:
        REDIS_CLIENT.set(f"session:{session_id}", pickle.dumps(messages), ex=86400)
    else:
        SESSIONS[session_id] = messages


# ─────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    session_id: Optional[str] = Field(default="default", description="Session UUID")

class CitationItem(BaseModel):
    citation_id: int
    source: str
    page: int
    claim: str
    snippet: str
    grounded: bool
    confidence_score: float

class RetrievalDiagnosticItem(BaseModel):
    citation_id: int
    source: str
    page: int
    dense_rank: Optional[int]
    bm25_rank: Optional[int]
    rrf_score: Optional[float]
    cross_encoder_score: Optional[float]
    snippet: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[str] = []
    citations: List[CitationItem] = []
    groundedness_score: float = 1.0
    retrieval_diagnostics: List[RetrievalDiagnosticItem] = []
    tool_calls_made: List[str] = []

class HealthResponse(BaseModel):
    status: str
    vectorstore_loaded: bool
    total_chunks: int

class VerifyCitationRequest(BaseModel):
    answer: str
    context_texts: List[str]


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    retriever = get_hybrid_retriever()
    chunks_path = VECTORSTORE_PATH / "chunks.pkl"
    total_chunks = 0
    if chunks_path.exists():
        try:
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            total_chunks = len(chunks)
        except Exception:
            pass

    return HealthResponse(
        status="ok",
        vectorstore_loaded=retriever is not None,
        total_chunks=total_chunks,
    )


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, and .md files are supported.")

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        res = run_ingestion_pipeline(file_path, VECTORSTORE_PATH)

        # Invalidate retriever singleton and reinitialize
        agent_module._hybrid_retriever_singleton = None
        init_retriever()

        return {
            "filename": file.filename,
            "chunks_added": res["chunks_added"],
            "total_chunks": res["total_chunks"],
            "status": "success",
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    graph = get_agent_graph()
    raw_history = get_session_history(request.session_id)

    clean_history = [
        msg for msg in raw_history
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    result = graph.invoke({
        "messages": clean_history + [HumanMessage(content=request.question)]
    })

    messages = result["messages"]
    final_message = messages[-1]
    citations = result.get("citations", [])
    groundedness_score = result.get("groundedness_score", 1.0)
    diagnostics = result.get("retrieval_diagnostics", [])

    sources = list({c["source"] for c in citations if "source" in c})
    tool_calls_made = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls_made.extend([tc["name"] for tc in msg.tool_calls])

    save_session_history(request.session_id, messages[-20:])

    return ChatResponse(
        question=request.question,
        answer=final_message.content,
        sources=sources,
        citations=citations,
        groundedness_score=groundedness_score,
        retrieval_diagnostics=diagnostics,
        tool_calls_made=tool_calls_made,
    )


async def agent_event_stream(question: str, session_id: str) -> AsyncGenerator[str, None]:
    graph = get_agent_graph()
    raw_history = get_session_history(session_id)
    clean_history = [
        msg for msg in raw_history
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    try:
        final_answer = ""
        citations = []
        groundedness_score = 1.0
        diagnostics = []

        all_messages = clean_history + [HumanMessage(content=question)]

        async for step in graph.astream({"messages": all_messages}):
            for node_name, node_output in step.items():
                if "citations" in node_output and node_output["citations"]:
                    citations = node_output["citations"]
                    groundedness_score = node_output.get("groundedness_score", 1.0)
                    diagnostics = node_output.get("retrieval_diagnostics", [])

                    yield f"data: {json.dumps({'type': 'citations', 'citations': citations, 'groundedness_score': groundedness_score})}\n\n"
                    yield f"data: {json.dumps({'type': 'diagnostics', 'diagnostics': diagnostics})}\n\n"

                for msg in node_output.get("messages", []):
                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                event = {"type": "tool_call", "tool": tc["name"], "args": tc["args"]}
                                yield f"data: {json.dumps(event)}\n\n"
                        else:
                            final_answer = msg.content
                            event = {"type": "answer", "content": msg.content}
                            yield f"data: {json.dumps(event)}\n\n"

                    elif isinstance(msg, ToolMessage):
                        event = {"type": "tool_result", "tool": msg.name, "result_preview": str(msg.content)[:200]}
                        yield f"data: {json.dumps(event)}\n\n"

        if final_answer:
            raw_history.append(HumanMessage(content=question))
            raw_history.append(AIMessage(content=final_answer))
            save_session_history(session_id, raw_history[-20:])

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        agent_event_stream(request.question, request.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/retrieval/inspect")
async def inspect_retrieval(query: str = Query(..., description="Query to inspect hybrid search scores")):
    retriever = get_hybrid_retriever()
    if retriever is None:
        raise HTTPException(status_code=400, detail="Retriever not initialized.")

    scored_results = retriever.get_relevant_documents_with_scores(query)
    diagnostics = []
    for doc, meta in scored_results:
        diagnostics.append({
            "citation_id": meta.get("final_rank"),
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", 1),
            "dense_rank": meta.get("dense_rank"),
            "bm25_rank": meta.get("bm25_rank"),
            "rrf_score": meta.get("rrf_score"),
            "cross_encoder_score": meta.get("cross_encoder_score"),
            "content_preview": doc.page_content[:200] + "...",
        })

    return {"query": query, "total_retrieved": len(diagnostics), "diagnostics": diagnostics}


@app.get("/sources")
async def list_sources():
    chunks_path = VECTORSTORE_PATH / "chunks.pkl"
    if not chunks_path.exists():
        return {"sources": [], "total_chunks": 0}

    try:
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        sources = sorted({c.metadata.get("source", "unknown") for c in chunks})
        return {"sources": sources, "total_chunks": len(chunks)}
    except Exception:
        return {"sources": [], "total_chunks": 0}


@app.delete("/sources/{filename}")
async def delete_source(filename: str):
    try:
        res = delete_document_from_stores(filename, VECTORSTORE_PATH)
        agent_module._hybrid_retriever_singleton = None
        init_retriever()
        return {
            "status": "success",
            "filename": filename,
            "details": res
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if REDIS_CLIENT:
        REDIS_CLIENT.delete(f"session:{session_id}")
    else:
        SESSIONS.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 60)
    print("  🚀 NeuRAG Production API Starting Up")
    print("=" * 60)
    get_embedding_model()
    init_retriever()
    print("  ✅ Backend ready at http://localhost:8000")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

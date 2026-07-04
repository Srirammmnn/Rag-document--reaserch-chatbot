"""
Phase 4: FastAPI Backend
=========================
Exposes the Phase 1-3 pipeline (ingestion + RAG + LangGraph agent) as a REST API.

Endpoints:
  POST /ingest         -> upload a PDF/TXT, runs Phase 1 ingestion pipeline
  POST /chat           -> ask a question, returns full JSON answer (blocking)
  POST /chat/stream     -> ask a question, streams the agent's reasoning via SSE
  GET  /health          -> liveness check
  GET  /sources         -> list ingested documents currently in the vectorstore

Run with:
  uvicorn main:app --reload --port 8000
"""

import os
import sys
import json

# Force UTF-8 output for Windows terminals to avoid emoji printing crashes
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import shutil
import asyncio
from pathlib import Path
from typing import List, Optional, AsyncGenerator

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# LangChain / LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from dotenv import load_dotenv
load_dotenv()

# Import the Phase 3 agent graph builder
# (In your real project structure, this would be: from agent import build_agent_graph)
# sys.path logic removed as files are now in a single directory


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="AI Research Assistant API",
    description="RAG + LangGraph agent backend — Phase 4 of the AI/ML learning project (updated)",
    version="1.0.0",
)

# CORS — required so Streamlit (different port) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTORSTORE_PATH = str(Path(__file__).parent / "vectorstore")
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Module-level singletons — loaded once, reused across requests
_embeddings = None
_vectorstore = None
_agent_graph = None


# ─────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS (Pydantic)
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    session_id: Optional[str] = Field(default="default", description="Conversation session ID")


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[str] = []
    tool_calls_made: List[str] = []


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    total_vectors: int
    status: str


class HealthResponse(BaseModel):
    status: str
    vectorstore_loaded: bool
    total_vectors: int


# ─────────────────────────────────────────────
# LAZY-LOADED SINGLETONS
# ─────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """Load embedding model once, reuse across all requests."""
    global _embeddings
    if _embeddings is None:
        print("🤖 Loading embedding model (first request only)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore() -> Optional[Pinecone]:
    """Connect to Pinecone cloud vector database."""
    global _vectorstore
    if _vectorstore is None:
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        index_name = os.environ.get("PINECONE_INDEX_NAME")
        if not pinecone_api_key or not index_name:
            print("⚠️ Pinecone credentials missing. Add PINECONE_API_KEY and PINECONE_INDEX_NAME to .env")
            return None
            
        print("📂 Connecting to Pinecone vector store...")
        try:
            _vectorstore = Pinecone(index_name=index_name, embedding=get_embeddings())
        except Exception as e:
            print(f"⚠️ Pinecone connection failed: {e}")
            return None
    return _vectorstore


def get_agent_graph():
    """
    Build (or reuse) the compiled LangGraph agent.
    This is the SAME graph from Phase 3 — wrapped here for API use.
    """
    global _agent_graph
    if _agent_graph is None:
        print("🏗️  Building agent graph...")
        from agent import build_agent_graph  # Phase 3 module
        _agent_graph = build_agent_graph()
    return _agent_graph


# ─────────────────────────────────────────────
# IN-MEMORY SESSION STORE (for conversation history)
# ─────────────────────────────────────────────
# Production note: replace with Redis or a DB for multi-instance deployments.

SESSIONS: dict = {}  # {session_id: List[BaseMessage]}

def get_session_history(session_id: str) -> list:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = []
    return SESSIONS[session_id]

# ─────────────────────────────────────────────
# SEMANTIC CACHE
# (Cleared on server reload)
# ─────────────────────────────────────────────

SEMANTIC_CACHE = []  # List of tuples: (query_embedding, final_answer)

def check_semantic_cache(question: str, threshold: float = 0.98):
    """Check if a similar query exists in cache and return its answer."""
    if not SEMANTIC_CACHE: return None
    
    emb_model = get_embeddings()
    query_emb = emb_model.embed_query(question)
    
    max_sim = -1.0
    best_ans = None
    
    for cached_emb, ans in SEMANTIC_CACHE:
        # dot product (vectors are normalized)
        sim = sum(a * b for a, b in zip(query_emb, cached_emb))
        if sim > max_sim:
            max_sim = sim
            best_ans = ans
            
    if max_sim >= threshold:
        print(f"⚡ Semantic cache hit! Similarity: {max_sim:.3f}")
        return best_ans
    return None

def add_to_semantic_cache(question: str, answer: str):
    emb_model = get_embeddings()
    query_emb = emb_model.embed_query(question)
    SEMANTIC_CACHE.append((query_emb, answer))
    if len(SEMANTIC_CACHE) > 200:
        SEMANTIC_CACHE.pop(0)


# ─────────────────────────────────────────────
# ENDPOINT 1: HEALTH CHECK
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness check — also reports whether the knowledge base is ready."""
    vs = get_vectorstore()
    return HealthResponse(
        status="ok",
        vectorstore_loaded=vs is not None,
        total_vectors=-1, # Pinecone does not expose local ntotal
    )


# ─────────────────────────────────────────────
# ENDPOINT 2: INGEST A DOCUMENT
# ─────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT file.

    Full pipeline for every upload:
      1. Parse & chunk the document
      2. Remove any old vectors for this filename from Pinecone (safe re-upload)
      3. Embed & push new chunks to Pinecone
      4. Deduplicate & save chunks.pkl (for BM25)
      5. Rebuild the in-memory BM25 + Pinecone retriever singleton instantly
         so RAG questions work immediately — no server restart needed
    """
    global _vectorstore

    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")

    # ── Keep a permanent copy so files survive server restarts ──
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # ── 1. Load & Split ──
        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        new_chunks = splitter.split_documents(documents)
        for chunk in new_chunks:
            chunk.metadata["source"] = file.filename

        if not new_chunks:
            raise HTTPException(status_code=400, detail="Document produced 0 chunks. Is it empty?")

        print(f"  📄 Loaded '{file.filename}' → {len(new_chunks)} chunks")

        # ── 2. Pinecone: delete old vectors for this source, then add new ──
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        index_name = os.environ.get("PINECONE_INDEX_NAME")
        if not pinecone_api_key or not index_name:
            raise HTTPException(status_code=500, detail="Pinecone credentials missing in .env")

        embeddings = get_embeddings()

        # Try to delete old vectors for this source (metadata filter)
        try:
            from pinecone import Pinecone as PineconeClient
            pc = PineconeClient(api_key=pinecone_api_key)
            idx = pc.Index(index_name)
            idx.delete(filter={"source": file.filename})
            print(f"  🗑️  Cleared old Pinecone vectors for '{file.filename}'")
        except Exception as del_err:
            print(f"  ⚠️  Could not delete old Pinecone vectors (OK on first upload): {del_err}")

        # Upload new vectors
        vs = Pinecone(index_name=index_name, embedding=embeddings)
        vs.add_documents(new_chunks)
        _vectorstore = vs
        print(f"  ☁️  Pushed {len(new_chunks)} vectors to Pinecone")

        # ── 3. chunks.pkl: remove old entries for this source, add new ──
        import pickle
        chunks_path = Path(VECTORSTORE_PATH) / "chunks.pkl"
        chunks_path.parent.mkdir(parents=True, exist_ok=True)

        existing_chunks = []
        if chunks_path.exists():
            with open(chunks_path, "rb") as f:
                try:
                    existing_chunks = pickle.load(f)
                except Exception:
                    existing_chunks = []

        # Remove stale chunks from the same file
        existing_chunks = [c for c in existing_chunks if c.metadata.get("source") != file.filename]
        existing_chunks.extend(new_chunks)

        with open(chunks_path, "wb") as f:
            pickle.dump(existing_chunks, f)
        print(f"  💾 chunks.pkl updated → {len(existing_chunks)} total chunks across all docs")

        # ── 4. Rebuild BM25 + Retriever singleton immediately ──
        # This is what makes new docs instantly searchable without a server restart.
        import agent as agent_module
        agent_module._hybrid_retriever = None   # invalidate stale singleton
        agent_module.init_retriever()           # rebuild from fresh chunks.pkl
        print(f"  ✅ Retriever singleton rebuilt — '{file.filename}' is now searchable!")

        # ── 5. Invalidate semantic cache (old answers may be stale) ──
        global SEMANTIC_CACHE
        SEMANTIC_CACHE.clear()
        print("  🧹 Semantic cache cleared")

        # Count unique sources now in the store
        sources_in_store = list({c.metadata.get("source", "?") for c in existing_chunks})

        return IngestResponse(
            filename=file.filename,
            chunks_added=len(new_chunks),
            total_vectors=len(existing_chunks),
            status=f"success — {len(sources_in_store)} document(s) in knowledge base: {sources_in_store}",
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    # NOTE: We intentionally do NOT delete file_path here — keep it in uploads/
    # so the server can re-ingest on next startup if needed.




# ─────────────────────────────────────────────
# ENDPOINT 3: CHAT (BLOCKING — full response at once)
# ─────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Ask the agent a question. Waits for the full agent loop to complete
    (including any tool calls) before returning the final answer.

    Use this when you don't need real-time streaming (e.g. programmatic API calls).
    """
    if get_vectorstore() is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Call POST /ingest first."
        )
        
    cached_answer = check_semantic_cache(request.question)
    if cached_answer:
        return ChatResponse(
            question=request.question,
            answer=cached_answer,
            sources=["semantic_cache"],
            tool_calls_made=[],
        )

    graph = get_agent_graph()
    raw_history = get_session_history(request.session_id)

    # Only pass clean Human+AI turns — strip ToolMessages and synthetic AIMessages
    # that carry tool_calls. These confuse the rag_node message index lookup.
    clean_history = [
        msg for msg in raw_history
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    # Run the LangGraph agent loop
    result = graph.invoke({
        "messages": clean_history + [HumanMessage(content=request.question)]
    })

    messages = result["messages"]
    final_message = messages[-1]

    # Extract which tools were called during this run (for transparency)
    tool_calls_made = []
    sources = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls_made.extend([tc["name"] for tc in msg.tool_calls])
        if isinstance(msg, ToolMessage) and msg.name == "search_knowledge_base":
            # crude source extraction from the tool output text
            for line in str(msg.content).split("\n"):
                if "(source:" in line:
                    src = line.split("(source:")[1].split(")")[0].strip()
                    sources.add(src)

    # Update session history (keep last 10 turns)
    SESSIONS[request.session_id] = messages[-20:]
    
    # Save to semantic cache only if the route is RAG or general LLM
    # (math, python, and web results should always execute dynamically)
    route = result.get("route")
    if route in {"rag", "llm"}:
        add_to_semantic_cache(request.question, final_message.content)
        print(f"  💾 Saved query to semantic cache (route: {route})")

    return ChatResponse(
        question=request.question,
        answer=final_message.content,
        sources=list(sources),
        tool_calls_made=tool_calls_made,
    )


# ─────────────────────────────────────────────
# ENDPOINT 4: CHAT STREAMING (SSE)
# ─────────────────────────────────────────────

async def agent_event_stream(question: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    Generator that yields Server-Sent Events as the agent executes.

    SSE format: each event is a line "data: {json}\\n\\n"
    The client (Streamlit/JS) reads these incrementally and updates the UI live.

    Event types emitted:
      "tool_call"  -> agent decided to call a tool (shown as "thinking..." in UI)
      "tool_result"-> a tool finished executing
      "token"      -> a chunk of the final answer (if using token streaming)
      "done"       -> final answer is complete
      "error"      -> something went wrong
    """
    graph = get_agent_graph()
    raw_history = get_session_history(session_id)
    clean_history = [
        msg for msg in raw_history
        if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None)
    ]

    try:
        final_answer = ""
        route_destination = None
        all_messages = clean_history + [HumanMessage(content=question)]

        async for step in graph.astream({"messages": all_messages}):
            for node_name, node_output in step.items():
                if node_name == "router":
                    route_destination = node_output.get("route")
                for msg in node_output["messages"]:

                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                event = {
                                    "type": "tool_call",
                                    "tool": tc["name"],
                                    "args": tc["args"],
                                }
                                yield f"data: {json.dumps(event)}\n\n"
                        else:
                            final_answer = msg.content
                            event = {"type": "answer", "content": msg.content}
                            yield f"data: {json.dumps(event)}\n\n"

                    elif isinstance(msg, ToolMessage):
                        event = {
                            "type": "tool_result",
                            "tool": msg.name,
                            "result_preview": str(msg.content)[:200],
                        }
                        yield f"data: {json.dumps(event)}\n\n"

        # Update session history
        if final_answer:
            raw_history.append(HumanMessage(content=question))
            raw_history.append(AIMessage(content=final_answer))
            if len(raw_history) > 20:
                raw_history[:] = raw_history[-20:]
            
            # Save to semantic cache only if the route is RAG or general LLM
            if route_destination in {"rag", "llm"}:
                add_to_semantic_cache(question, final_answer)
                print(f"  💾 Saved streaming query to semantic cache (route: {route_destination})")

        # Signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming version of /chat using Server-Sent Events (SSE).

    The client receives events as the agent works:
      1. {"type": "tool_call", "tool": "search_knowledge_base", ...}  <- agent decided to search
      2. {"type": "tool_result", "tool": "search_knowledge_base", ...} <- search completed
      3. {"type": "answer", "content": "RAG stands for..."}            <- final answer
      4. {"type": "done"}                                               <- stream complete

    This lets the UI show "Searching knowledge base..." then "Thinking..."
    then stream in the final answer — much better UX than waiting in silence.
    """
    if get_vectorstore() is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Call POST /ingest first."
        )
        
    cached_answer = check_semantic_cache(request.question)
    if cached_answer:
        async def cached_stream():
            yield f"data: {json.dumps({'type': 'answer', 'content': cached_answer})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        # Update session history even for cached responses
        raw_history = get_session_history(request.session_id)
        raw_history.append(HumanMessage(content=request.question))
        raw_history.append(AIMessage(content=cached_answer))
        if len(raw_history) > 20:
            raw_history[:] = raw_history[-20:]
            
        return StreamingResponse(
            cached_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        agent_event_stream(request.question, request.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering for real-time streaming
        },
    )


# ─────────────────────────────────────────────
# ENDPOINT 5: LIST SOURCES
# ─────────────────────────────────────────────

@app.get("/sources")
async def list_sources():
    """List the unique document sources currently in the knowledge base."""
    import pickle
    chunks_path = Path(VECTORSTORE_PATH) / "chunks.pkl"
    if not chunks_path.exists():
        return {"sources": [], "total_chunks": 0}

    try:
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        sources = sorted({c.metadata.get("source", "unknown") for c in chunks})
        return {"sources": sources, "total_chunks": len(chunks)}
    except Exception:
        return {"sources": [], "total_chunks": 0}


# ─────────────────────────────────────────────
# ENDPOINT 6: CLEAR SESSION
# ─────────────────────────────────────────────

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session (start fresh)."""
    SESSIONS.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


# ─────────────────────────────────────────────
# STARTUP EVENT — preload models so first request isn't slow
# ─────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 60)
    print("  🚀 AI Research Assistant API starting up")
    print("=" * 60)
    get_embeddings()       # preload embedding model
    get_vectorstore()      # preload vectorstore if it exists

    # Preload cross encoder + retriever singleton
    from agent import load_cross_encoder, init_retriever
    load_cross_encoder()
    init_retriever()   # build BM25 + connect Pinecone ONCE here, not per-request

    # ── Auto-restore: re-ingest any files in uploads/ that are missing from chunks.pkl ──
    # This ensures documents survive server restarts without manual re-upload.
    import pickle
    chunks_path = Path(VECTORSTORE_PATH) / "chunks.pkl"
    already_indexed = set()
    if chunks_path.exists():
        try:
            with open(chunks_path, "rb") as f:
                existing = pickle.load(f)
            already_indexed = {c.metadata.get("source") for c in existing}
        except Exception:
            pass

    uploads = list(UPLOAD_DIR.glob("*.pdf")) + list(UPLOAD_DIR.glob("*.txt"))
    missing = [u for u in uploads if u.name not in already_indexed]
    if missing:
        print(f"  🔄 Re-ingesting {len(missing)} file(s) missing from knowledge base: {[u.name for u in missing]}")
        from fastapi.datastructures import UploadFile as FUploadFile
        import io
        for fpath in missing:
            try:
                with open(fpath, "rb") as raw:
                    content = raw.read()
                mock_file = UploadFile(
                    filename=fpath.name,
                    file=io.BytesIO(content),
                )
                await ingest_document(mock_file)
                print(f"    ✅ Re-ingested: {fpath.name}")
            except Exception as e:
                print(f"    ⚠️ Failed to re-ingest {fpath.name}: {e}")
    else:
        print("  ✅ All uploaded files are already indexed")

    print("  ✅ Ready to accept requests")
    print("  📚 Docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

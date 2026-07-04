"""
Phase 1: Document Ingestion Pipeline
=====================================
Flow: PDF/TXT/URL → Loader → Text Splitter → Embedding Model → FAISS Index

Key concepts covered:
  - Document loaders (PDF, text, web)
  - Text chunking strategies
  - Embeddings (what they are + how to generate)
  - Vector stores (FAISS) - storing & searching embeddings
  - Similarity search (cosine distance)
"""

import os
import pickle
from pathlib import Path
from typing import List

# LangChain document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,          # loads PDF files, one Document per page
    TextLoader,           # loads plain .txt files
    WebBaseLoader,        # scrapes a URL and loads its text
    DirectoryLoader,      # loads all files in a folder
)

# Text splitters — break large docs into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter is the best default:
# it tries to split on ["\n\n", "\n", " ", ""] in order,
# so it preserves paragraph → sentence → word structure

# Embeddings — convert text into dense numerical vectors
from langchain_huggingface import HuggingFaceEmbeddings
# HuggingFace runs LOCALLY (no API key needed, no cost)
# "all-MiniLM-L6-v2" is 384-dimensional, fast, and very good for retrieval

# Vector store — stores embeddings + enables similarity search
from langchain_community.vectorstores import FAISS
# FAISS (Facebook AI Similarity Search) = fast nearest-neighbor search
# It stores vectors in RAM and searches using cosine/L2 distance

from langchain_core.documents import Document  # the core LangChain data unit


# ─────────────────────────────────────────────
# STEP 1: LOAD DOCUMENTS
# ─────────────────────────────────────────────

def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file.
    Each page becomes one Document with metadata: {source, page}.
    """
    print(f"  📄 Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"     → Loaded {len(documents)} pages")
    return documents


def load_text(file_path: str) -> List[Document]:
    """
    Load a plain text file.
    The whole file becomes one Document.
    """
    print(f"  📝 Loading text file: {file_path}")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    print(f"     → Loaded {len(documents)} document(s)")
    return documents


def load_url(url: str) -> List[Document]:
    """
    Scrape a URL and load its text content.
    Great for loading documentation pages, blog posts, etc.
    """
    print(f"  🌐 Loading URL: {url}")
    loader = WebBaseLoader(url)
    documents = loader.load()
    print(f"     → Loaded {len(documents)} page(s) from web")
    return documents


def load_directory(dir_path: str, glob_pattern: str = "**/*.pdf") -> List[Document]:
    """
    Load all matching files from a directory.
    glob_pattern examples: "**/*.pdf", "**/*.txt", "**/*.md"
    """
    print(f"  📁 Loading directory: {dir_path} (pattern: {glob_pattern})")
    loader = DirectoryLoader(
        dir_path,
        glob=glob_pattern,
        loader_cls=PyPDFLoader if "pdf" in glob_pattern else TextLoader,
    )
    documents = loader.load()
    print(f"     → Loaded {len(documents)} document(s)")
    return documents


def create_sample_documents() -> List[Document]:
    """
    Create sample in-memory Documents for testing (no files needed).
    In real usage you'd use the loaders above.
    Document has two fields:
      - page_content: the actual text
      - metadata: a dict with source info (source, page, author, etc.)
    """
    docs = [
        Document(
            page_content="""
            Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs
            by retrieving relevant context from an external knowledge base before
            generating a response. RAG solves the key limitation of LLMs: static
            training data. With RAG, you can give the model access to up-to-date
            or private information without retraining it.
            The two main phases are: (1) offline ingestion - chunking and embedding
            documents into a vector store, and (2) online retrieval - searching for
            relevant chunks at query time and passing them as context to the LLM.
            """,
            metadata={"source": "rag_overview.txt", "topic": "RAG", "page": 1}
        ),
        Document(
            page_content="""
            Vector embeddings are dense numerical representations of text. A sentence
            like "The cat sat on the mat" becomes a list of 384 numbers (in MiniLM)
            where semantically similar sentences have vectors that are close together
            in high-dimensional space. This is measured using cosine similarity:
            sim(A, B) = (A · B) / (||A|| × ||B||). Values range from -1 to 1, where
            1 means identical meaning, 0 means unrelated, -1 means opposite.
            Embeddings capture semantic meaning, not just keyword overlap, which is
            why "fast car" and "speedy automobile" will have similar vectors.
            """,
            metadata={"source": "embeddings_guide.txt", "topic": "Embeddings", "page": 1}
        ),
        Document(
            page_content="""
            FAISS (Facebook AI Similarity Search) is a library for efficient similarity
            search over dense vectors. It supports both exact and approximate nearest
            neighbor search. The two most common index types are:
            - IndexFlatL2: exact search using L2 (Euclidean) distance. Slow but precise.
            - IndexIVFFlat: approximate search using inverted file index. Fast but slightly less accurate.
            For most RAG use cases, FAISS IndexFlatL2 (the default in LangChain) is
            fine up to ~100k documents. Beyond that, consider Qdrant or Pinecone.
            """,
            metadata={"source": "faiss_guide.txt", "topic": "Vector DB", "page": 1}
        ),
        Document(
            page_content="""
            LangGraph is a framework built on top of LangChain for building stateful,
            multi-step AI agent workflows. It models your agent as a directed graph
            where nodes are processing functions and edges define the flow between them.
            Key concepts: StateGraph (the graph itself), AgentState (shared state dict
            passed between nodes), add_node (register a function as a node), add_edge
            (connect two nodes), add_conditional_edges (branching logic based on state).
            LangGraph is ideal for building ReAct agents, multi-agent systems, and
            any workflow that requires loops, branching, or human-in-the-loop steps.
            """,
            metadata={"source": "langgraph_intro.txt", "topic": "LangGraph", "page": 1}
        ),
        Document(
            page_content="""
            Text chunking is the process of splitting large documents into smaller
            pieces before embedding them. This matters because:
            1. LLMs have context window limits (you can't fit a 500-page PDF)
            2. Smaller chunks = more precise retrieval (you retrieve the exact paragraph)
            3. Embedding models work best on short passages (< 512 tokens)
            The RecursiveCharacterTextSplitter splits on [paragraph, newline, space]
            in order, trying to keep semantic units together. Key parameters:
            - chunk_size: max tokens per chunk (500-1000 is common)
            - chunk_overlap: overlap between chunks (50-200) to avoid losing context
              at boundaries. Think of it as a sliding window.
            """,
            metadata={"source": "chunking_guide.txt", "topic": "Text Splitting", "page": 1}
        ),
    ]
    print(f"  ✅ Created {len(docs)} sample in-memory documents")
    return docs


# ─────────────────────────────────────────────
# STEP 2: SPLIT INTO CHUNKS
# ─────────────────────────────────────────────

def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.

    Why chunk?
      - Embedding models have a max input length (~512 tokens)
      - Smaller chunks = more precise retrieval
      - A 10-page PDF becomes ~50 focused chunks, each independently searchable

    chunk_size:    target size in characters (not tokens — be aware of this!)
    chunk_overlap: how much consecutive chunks share (prevents losing context at edges)

    Example:
      chunk_size=500, chunk_overlap=100 means:
        chunk 1: chars 0-500
        chunk 2: chars 400-900  ← 100 char overlap with chunk 1
        chunk 3: chars 800-1300 ← 100 char overlap with chunk 2
    """
    print(f"\n✂️  Splitting {len(documents)} documents...")
    print(f"   chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # These separators are tried in order — splits on paragraphs first,
        # then newlines, then spaces, then chars as last resort
        separators=["\n\n", "\n", ". ", " ", ""],
        # length_function=len uses character count
        # Use tiktoken for token-accurate splitting:
        # length_function=tiktoken_len  (see commented code below)
    )

    chunks = splitter.split_documents(documents)

    print(f"   → {len(documents)} docs became {len(chunks)} chunks")
    print(f"   → Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    # Each chunk inherits the parent document's metadata
    # You can also add chunk-specific metadata here:
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    return chunks


# ─────────────────────────────────────────────
# STEP 3: CREATE EMBEDDINGS
# ─────────────────────────────────────────────

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Load a HuggingFace embedding model.

    Model options (all free, run locally):
      "all-MiniLM-L6-v2"      → 384 dims, 80MB, very fast — BEST for starters
      "all-mpnet-base-v2"     → 768 dims, 420MB, higher quality
      "BAAI/bge-small-en-v1.5"→ 384 dims, optimized for retrieval (state-of-art small)
      "BAAI/bge-large-en-v1.5"→ 1024 dims, best quality, larger model

    For production, OpenAI's text-embedding-3-small (1536 dims) is excellent
    but costs money per token.

    First run: downloads model from HuggingFace hub (~80MB for MiniLM)
    Subsequent runs: loads from cache instantly
    """
    print(f"\n🤖 Loading embedding model: {model_name}")
    print(f"   (First run downloads the model — ~80MB for MiniLM)")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},   # use "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True},
        # normalize_embeddings=True → cosine similarity = dot product (faster search)
    )

    # Demo: show what an embedding looks like
    sample_text = "What is RAG?"
    sample_vector = embeddings.embed_query(sample_text)
    print(f"   → Model loaded!")
    print(f"   → Embedding dims: {len(sample_vector)}")
    print(f"   → Sample vector (first 5 of {len(sample_vector)}): {[round(x, 4) for x in sample_vector[:5]]}")

    return embeddings


# ─────────────────────────────────────────────
# STEP 4: BUILD FAISS VECTOR STORE
# ─────────────────────────────────────────────

def build_vectorstore(chunks: List[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Embed all chunks and store them in FAISS.

    What happens internally:
      1. Each chunk's text is passed through the embedding model
      2. The resulting vector (384 floats for MiniLM) is stored in FAISS
      3. FAISS builds an index for fast nearest-neighbor lookup
      4. The original text + metadata is stored alongside the vectors

    Memory: ~384 floats × 4 bytes × num_chunks
    e.g., 1000 chunks = ~1.5MB (very small!)
    """
    print(f"\n🗄️  Building FAISS vector store from {len(chunks)} chunks...")
    print(f"   Embedding all chunks (this may take 10-30 seconds)...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
        # FAISS uses IndexFlatL2 by default = exact search, no approximation
        # For >100k docs, switch to FAISS with IVF index for speed
    )

    total_vectors = vectorstore.index.ntotal
    vector_dim = vectorstore.index.d
    print(f"   → Vector store built!")
    print(f"   → Total vectors stored: {total_vectors}")
    print(f"   → Vector dimensions: {vector_dim}")

    return vectorstore


# ─────────────────────────────────────────────
# STEP 5: PERSIST TO DISK
# ─────────────────────────────────────────────

def save_vectorstore(vectorstore: FAISS, chunks: List[Document], save_path: str = "vectorstore") -> None:
    """
    Save the FAISS index to disk so you don't re-embed on every restart.
    Saves three files:
      - {save_path}/index.faiss  → the FAISS binary index
      - {save_path}/index.pkl    → the docstore (texts + metadata)
      - {save_path}/chunks.pkl   → the raw chunks (needed for BM25 sparse search)
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(save_path)
    
    # Save the raw chunks for BM25 Retriever
    with open(Path(save_path) / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
        
    print(f"\n💾 Vector store saved to: {save_path}/")
    print(f"   Files: index.faiss + index.pkl + chunks.pkl")


def load_vectorstore(save_path: str, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Load a previously saved FAISS index from disk.
    Much faster than re-embedding — use this in Phase 2 for the RAG chain.
    """
    print(f"\n📂 Loading vector store from: {save_path}/")
    vectorstore = FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True  # required in newer LangChain versions
    )
    print(f"   → Loaded {vectorstore.index.ntotal} vectors")
    return vectorstore


# ─────────────────────────────────────────────
# STEP 6: SEARCH THE VECTOR STORE
# ─────────────────────────────────────────────

def similarity_search(vectorstore: FAISS, query: str, k: int = 3) -> None:
    """
    Search the vector store for the k most similar chunks to the query.

    What happens internally:
      1. Query text → embedding model → query vector (384 floats)
      2. FAISS computes L2 distance between query vector and all stored vectors
      3. Returns the k chunks with smallest distance (= most similar)

    The score is L2 distance (lower = more similar).
    Use similarity_search_with_score() to see the scores.
    """
    print(f"\n🔍 Query: '{query}'")
    print(f"   Searching for top {k} most similar chunks...\n")

    # Basic search — returns List[Document]
    results = vectorstore.similarity_search(query, k=k)

    # Search with scores — returns List[Tuple[Document, float]]
    results_with_scores = vectorstore.similarity_search_with_score(query, k=k)

    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"  Result {i} (L2 distance: {score:.4f} — lower is better)")
        print(f"  Source: {doc.metadata.get('source', 'unknown')} | Topic: {doc.metadata.get('topic', 'N/A')}")
        print(f"  Content: {doc.page_content.strip()[:200]}...")
        print()

    return results


def mmr_search(vectorstore: FAISS, query: str, k: int = 3) -> List[Document]:
    """
    MMR = Maximal Marginal Relevance search.
    Balances relevance WITH diversity — avoids returning 3 very similar chunks.
    Better than plain similarity_search when you want varied coverage.

    fetch_k: initially fetch more candidates, then pick k diverse ones
    lambda_mult: 0 = max diversity, 1 = max relevance (0.5 is balanced)
    """
    print(f"\n🔀 MMR Search: '{query}' (diverse results)")
    results = vectorstore.max_marginal_relevance_search(
        query, k=k, fetch_k=k * 3, lambda_mult=0.5
    )
    for i, doc in enumerate(results, 1):
        print(f"  MMR Result {i}: {doc.metadata.get('topic')} | {doc.page_content.strip()[:120]}...")
    return results


# ─────────────────────────────────────────────
# MAIN: RUN THE FULL PIPELINE
# ─────────────────────────────────────────────

def run_ingestion_pipeline(
    source_type: str = "sample",   # "sample" | "pdf" | "url" | "directory"
    source_path: str = None,
    vectorstore_path: str = str(Path(__file__).parent / "vectorstore"),
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> FAISS:
    """
    Full ingestion pipeline:
      load → split → embed → index → save → return vectorstore

    Args:
        source_type:      where to load docs from
        source_path:      file/dir/URL path (not needed for "sample")
        vectorstore_path: where to save the FAISS index
        chunk_size:       characters per chunk
        chunk_overlap:    overlap between consecutive chunks
    """
    print("=" * 60)
    print("  Phase 1: Document Ingestion Pipeline")
    print("=" * 60)

    # ── 1. LOAD ─────────────────────────────
    print("\n📥 STEP 1: Loading documents")
    if source_type == "sample":
        documents = create_sample_documents()
    elif source_type == "pdf":
        documents = load_pdf(source_path)
    elif source_type == "txt":
        documents = load_text(source_path)
    elif source_type == "url":
        documents = load_url(source_path)
    elif source_type == "directory":
        documents = load_directory(source_path)
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    # ── 2. SPLIT ─────────────────────────────
    print("\n✂️  STEP 2: Splitting into chunks")
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # ── 3. EMBED ─────────────────────────────
    print("\n🤖 STEP 3: Loading embedding model")
    embeddings = get_embedding_model("all-MiniLM-L6-v2")

    # ── 4. INDEX ─────────────────────────────
    print("\n🗄️  STEP 4: Building FAISS vector store")
    vectorstore = build_vectorstore(chunks, embeddings)

    # ── 5. SAVE ──────────────────────────────
    print("\n💾 STEP 5: Saving to disk")
    save_vectorstore(vectorstore, chunks, vectorstore_path)

    # ── 6. TEST SEARCH ───────────────────────
    print("\n🔍 STEP 6: Testing similarity search")
    print("-" * 40)
    similarity_search(vectorstore, "How does RAG work?", k=2)
    similarity_search(vectorstore, "What are vector embeddings?", k=2)
    mmr_search(vectorstore, "text splitting and chunking", k=2)

    print("\n" + "=" * 60)
    print("  ✅ Phase 1 Complete!")
    print("  Next: Phase 2 — Build the RAG Chain with this vectorstore")
    print("=" * 60)

    return vectorstore


# ─────────────────────────────────────────────
# BONUS: ADD NEW DOCS TO EXISTING INDEX
# ─────────────────────────────────────────────

def add_documents_to_existing(
    new_docs: List[Document],
    vectorstore_path: str,
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Add new documents to an already-built vector store without rebuilding.
    This is how you'd handle incremental updates in production.
    """
    print(f"\n➕ Adding {len(new_docs)} new documents to existing vectorstore...")
    vectorstore = load_vectorstore(vectorstore_path, embeddings)

    chunks = split_documents(new_docs)
    vectorstore.add_documents(chunks)

    save_vectorstore(vectorstore, chunks, vectorstore_path)
    print(f"   → New total: {vectorstore.index.ntotal} vectors")
    return vectorstore


if __name__ == "__main__":
    # ── Run with sample data (no files needed) ──
    vectorstore = run_ingestion_pipeline(source_type="sample")

    # ── To ingest a real PDF, uncomment: ──
    # vectorstore = run_ingestion_pipeline(
    #     source_type="pdf",
    #     source_path="your_paper.pdf"
    # )

    # ── To ingest a URL (e.g., LangChain docs), uncomment: ──
    # vectorstore = run_ingestion_pipeline(
    #     source_type="url",
    #     source_path="https://python.langchain.com/docs/introduction/"
    # )

    # ── To load a saved vectorstore in another script: ──
    # from ingest import load_vectorstore, get_embedding_model
    # embeddings = get_embedding_model()
    # vectorstore = load_vectorstore("vectorstore", embeddings)

"""
Phase 1 & Ingestion Module: Production Document Processing & Hybrid Indexing
=============================================================================
Flow: PDF/TXT/MD/URL -> Document Loaders -> Semantic Splitter -> Dense Store (Pinecone/Chroma/FAISS) + BM25

Features:
  - Preserves page numbers, source filenames, and section offsets.
  - Generates chunks.pkl for BM25 sparse keyword indexing.
  - Supports automatic fallback to local ChromaDB / FAISS if Pinecone API keys are absent.
"""

import os
import sys
import pickle
import shutil
from pathlib import Path
from typing import List, Optional

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

_embeddings_singleton = None

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """Singleton getter for HuggingFace embedding model."""
    global _embeddings_singleton
    if _embeddings_singleton is None:
        print(f"🤖 Loading HuggingFace embedding model: {model_name}...")
        _embeddings_singleton = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_singleton


def load_document_file(file_path: Path) -> List[Document]:
    """Load a PDF, TXT, or Markdown document."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif ext in [".txt", ".md"]:
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    docs = loader.load()
    if not docs:
        raise ValueError("The uploaded document is empty or could not be read.")
    return docs


def process_and_chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits documents into clean chunks while preserving metadata and adding chunk_id.
    Uses optimal 1000-char chunk size with 200-char overlap for production RAG context preservation.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks could be extracted from the document. Please ensure the file contains readable text.")

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(i)
        chunk.metadata["page"] = int(chunk.metadata.get("page", 1))
        if "source" in chunk.metadata:
            src_name = Path(chunk.metadata["source"]).name
            chunk.metadata["source"] = src_name
            chunk.page_content = f"Source Document: {src_name}\n\n{chunk.page_content}"

    return chunks


def index_documents_to_dense_store(
    chunks: List[Document],
    vectorstore_path: Path,
    index_name: Optional[str] = None
):
    """
    Pushes dense vectors to Pinecone if API keys are available;
    otherwise falls back to FAISS/ChromaDB locally.
    """
    embeddings = get_embedding_model()
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    pinecone_idx = os.environ.get("PINECONE_INDEX_NAME") or index_name or "rag"

    if pinecone_key and pinecone_idx:
        try:
            print(f"☁️ Indexing {len(chunks)} chunks into Pinecone cloud vectorstore (index: '{pinecone_idx}')...")
            from pinecone import Pinecone as PineconeClient, ServerlessSpec
            from langchain_pinecone import Pinecone

            pc = PineconeClient(api_key=pinecone_key)
            existing_indexes = pc.list_indexes().names()

            if pinecone_idx not in existing_indexes:
                print(f"  🔨 Creating Pinecone index '{pinecone_idx}' with 384 dimensions...")
                pc.create_index(
                    name=pinecone_idx,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                print(f"  ✅ Pinecone index '{pinecone_idx}' created successfully.")

            index_obj = pc.Index(pinecone_idx)
            vs = Pinecone(index=index_obj, embedding=embeddings)

            # Generate deterministic vector IDs to prevent duplicate vectors
            chunk_ids = [f"{c.metadata.get('source')}::chunk_{c.metadata.get('chunk_id')}" for c in chunks]
            vs.add_documents(chunks, ids=chunk_ids)
            print(f"  ✅ Pushed {len(chunks)} vectors to Pinecone index '{pinecone_idx}'")
            return vs
        except Exception as e:
            print(f"⚠️ Pinecone index failed: {e}. Falling back to local store.")

    # Local FAISS / ChromaDB Vector Store Fallback
    print("💾 Indexing chunks into local vectorstore...")
    
    # Try ChromaDB if available
    try:
        from langchain_chroma import Chroma
        chroma_dir = vectorstore_path / "chroma_db"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(chroma_dir)
        )
        print(f"  ✅ Saved {len(chunks)} vectors to local ChromaDB store.")
    except Exception as chroma_err:
        # Fallback to FAISS
        from langchain_community.vectorstores import FAISS
        faiss_dir = vectorstore_path / "faiss_index"
        faiss_dir.mkdir(parents=True, exist_ok=True)

        if (faiss_dir / "index.faiss").exists():
            try:
                vs = FAISS.load_local(str(faiss_dir), embeddings, allow_dangerous_deserialization=True)
                vs.add_documents(chunks)
            except Exception:
                vs = FAISS.from_documents(chunks, embeddings)
        else:
            vs = FAISS.from_documents(chunks, embeddings)

        vs.save_local(str(faiss_dir))
        print(f"  ✅ Saved {len(chunks)} vectors to local FAISS store.")

    return vs


def update_bm25_chunks_file(new_chunks: List[Document], vectorstore_path: Path):
    """Updates chunks.pkl used by BM25 sparse search."""
    chunks_path = vectorstore_path / "chunks.pkl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    existing_chunks = []
    if chunks_path.exists():
        try:
            with open(chunks_path, "rb") as f:
                existing_chunks = pickle.load(f)
        except Exception:
            existing_chunks = []

    # Filter out old chunks from the same file names
    new_sources = {c.metadata.get("source") for c in new_chunks}
    filtered_existing = [c for c in existing_chunks if c.metadata.get("source") not in new_sources]

    combined = filtered_existing + new_chunks
    with open(chunks_path, "wb") as f:
        pickle.dump(combined, f)

    print(f"💾 Updated BM25 index file chunks.pkl ({len(combined)} total chunks across all documents)")
    return len(combined)


def delete_document_from_stores(filename: str, vectorstore_path: Path) -> dict:
    """
    Deletes all vectors and metadata for a specific document filename from:
    1. Pinecone cloud index
    2. BM25 chunks.pkl
    3. uploads/ folder
    """
    print(f"🗑️ Deleting document: {filename}")
    results = {"filename": filename, "pinecone_deleted": False, "bm25_deleted": False, "file_deleted": False}

    # 1. Delete from Pinecone
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    pinecone_idx = os.environ.get("PINECONE_INDEX_NAME") or "rag"
    if pinecone_key and pinecone_idx:
        try:
            from pinecone import Pinecone as PineconeClient
            pc = PineconeClient(api_key=pinecone_key)
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            if pinecone_idx in existing_indexes:
                index = pc.Index(pinecone_idx)
                index.delete(filter={"source": filename})
                results["pinecone_deleted"] = True
                print(f"  ✅ Deleted vectors for '{filename}' from Pinecone index '{pinecone_idx}'")
        except Exception as e:
            print(f"  ⚠️ Pinecone deletion failed: {e}")

    # 2. Delete from chunks.pkl (BM25)
    chunks_path = vectorstore_path / "chunks.pkl"
    if chunks_path.exists():
        try:
            with open(chunks_path, "rb") as f:
                existing_chunks = pickle.load(f)

            initial_count = len(existing_chunks)
            filtered_chunks = [c for c in existing_chunks if c.metadata.get("source") != filename]

            with open(chunks_path, "wb") as f:
                pickle.dump(filtered_chunks, f)

            results["bm25_deleted"] = True
            results["chunks_removed"] = initial_count - len(filtered_chunks)
            results["remaining_chunks"] = len(filtered_chunks)
            print(f"  ✅ Removed {initial_count - len(filtered_chunks)} chunks from BM25 chunks.pkl ({len(filtered_chunks)} remaining)")
        except Exception as e:
            print(f"  ⚠️ BM25 chunk removal failed: {e}")

    # 3. Remove raw file from uploads/
    uploads_dir = vectorstore_path.parent / "uploads"
    file_path = uploads_dir / filename
    if file_path.exists():
        try:
            file_path.unlink()
            results["file_deleted"] = True
            print(f"  ✅ Deleted raw uploaded file: {filename}")
        except Exception as e:
            print(f"  ⚠️ Could not remove raw file '{filename}': {e}")

    return results


def run_ingestion_pipeline(file_path: Path, vectorstore_path: Path) -> dict:
    """Run full ingestion flow for a given file path."""
    print(f"🚀 Starting Ingestion for file: {file_path.name}")
    raw_docs = load_document_file(file_path)
    chunks = process_and_chunk_documents(raw_docs)

    index_documents_to_dense_store(chunks, vectorstore_path)
    total_bm25_chunks = update_bm25_chunks_file(chunks, vectorstore_path)

    return {
        "filename": file_path.name,
        "chunks_added": len(chunks),
        "total_chunks": total_bm25_chunks,
        "status": "success",
    }


if __name__ == "__main__":
    test_file = Path("uploads/sample.txt")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("RAG stands for Retrieval Augmented Generation. Hybrid Search uses Dense + Sparse vectors.")
    res = run_ingestion_pipeline(test_file, Path("vectorstore"))
    print(res)


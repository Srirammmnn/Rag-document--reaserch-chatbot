"""
Phase 1 & Ingestion Module: Production Document Processing & Dense Indexing
=============================================================================
Flow: PDF/TXT/MD/URL -> Document Loaders -> Semantic Splitter -> Pinecone

Features:
  - Preserves page numbers, source filenames, and section offsets.
  - Vercel-compatible (No local storage or heavy ML dependencies).
  - Uses Google Generative AI Embeddings.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

_embeddings_singleton = None

def get_embedding_model(model_name: str = "models/embedding-001") -> GoogleGenerativeAIEmbeddings:
    """Singleton getter for Google embedding model."""
    global _embeddings_singleton
    if _embeddings_singleton is None:
        print(f"🤖 Loading Google Generative AI embedding model: {model_name}...")
        _embeddings_singleton = GoogleGenerativeAIEmbeddings(model=model_name)
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
    Pushes dense vectors to Pinecone.
    """
    embeddings = get_embedding_model()
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    pinecone_idx = os.environ.get("PINECONE_INDEX_NAME") or index_name or "rag"

    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY is required for Vercel deployment.")

    print(f"☁️ Indexing {len(chunks)} chunks into Pinecone cloud vectorstore (index: '{pinecone_idx}')...")
    from pinecone import Pinecone as PineconeClient, ServerlessSpec
    from langchain_pinecone import Pinecone

    pc = PineconeClient(api_key=pinecone_key)
    existing_indexes = pc.list_indexes().names()

    if pinecone_idx not in existing_indexes:
        print(f"  🔨 Creating Pinecone index '{pinecone_idx}' with 768 dimensions (Google GenAI)...")
        pc.create_index(
            name=pinecone_idx,
            dimension=768, # Google embeddings are 768 dim
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


def delete_document_from_stores(filename: str, vectorstore_path: Path) -> dict:
    """
    Deletes all vectors and metadata for a specific document filename from:
    1. Pinecone cloud index
    2. uploads/ folder
    """
    print(f"🗑️ Deleting document: {filename}")
    results = {"filename": filename, "pinecone_deleted": False, "file_deleted": False}

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

    # 2. Remove raw file from uploads/
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

    return {
        "filename": file_path.name,
        "chunks_added": len(chunks),
        "total_chunks": len(chunks),
        "status": "success",
    }


if __name__ == "__main__":
    test_file = Path("uploads/sample.txt")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("RAG stands for Retrieval Augmented Generation. Vercel deployment requires cloud vectorstores.")
    res = run_ingestion_pipeline(test_file, Path("vectorstore"))
    print(res)

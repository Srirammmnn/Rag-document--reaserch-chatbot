import os
import shutil
from pathlib import Path
from ingest import run_ingestion_pipeline

def rebuild():
    base_dir = Path(__file__).parent
    uploads_dir = base_dir / "uploads"
    vector_dir = base_dir / "vectorstore"
    
    if not uploads_dir.exists():
        print("No uploads directory.")
        return
        
    # get files
    files = list(uploads_dir.glob("*.*"))
    print(f"Found {len(files)} files to re-ingest: {[f.name for f in files]}")
    
    # wipe vectorstore completely
    if vector_dir.exists():
        print("Wiping existing vectorstore...")
        shutil.rmtree(vector_dir)
        
    vector_dir.mkdir(parents=True, exist_ok=True)
    
    # re-ingest
    for f in files:
        if f.suffix.lower() in [".pdf", ".txt", ".md"]:
            print(f"Re-ingesting {f.name}...")
            run_ingestion_pipeline(f, vector_dir)
            
    print("Done. Please restart the backend to load the new vectorstore.")

if __name__ == "__main__":
    rebuild()

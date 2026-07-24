import os
import time
import requests
import tempfile
import shutil
from pathlib import Path

def reingest():
    base_dir = Path(__file__).parent
    uploads_dir = base_dir / "uploads"
    
    if not uploads_dir.exists():
        print("No uploads directory.")
        return
        
    files = list(uploads_dir.glob("*.*"))
    if not files:
        print("No files to process.")
        return

    # Backup files
    temp_dir = Path(tempfile.mkdtemp())
    backup_paths = []
    for f in files:
        if f.suffix.lower() in [".pdf", ".txt", ".md"]:
            dst = temp_dir / f.name
            shutil.copy2(f, dst)
            backup_paths.append(dst)
            
    print(f"Backed up {len(backup_paths)} files to {temp_dir}")
    
    for f_path in backup_paths:
        filename = f_path.name
        
        # 1. Delete from server (this wipes from pinecone, chunks.pkl, and deletes the local upload file)
        print(f"Deleting {filename} from server...")
        try:
            res = requests.delete(f"http://127.0.0.1:8000/sources/{filename}")
            print("Delete response:", res.status_code, res.text)
        except Exception as e:
            print(f"Failed to delete {filename}: {e}")
            
        time.sleep(1)
            
        # 2. Upload and re-ingest
        print(f"Re-ingesting {filename}...")
        try:
            with open(f_path, 'rb') as f:
                res = requests.post("http://127.0.0.1:8000/ingest", files={"file": (filename, f)})
                print("Ingest response:", res.status_code, res.text)
        except Exception as e:
            print(f"Failed to ingest {filename}: {e}")
            
        time.sleep(1)
        
    # Cleanup temp
    shutil.rmtree(temp_dir)
    print("Done reingesting via API!")

if __name__ == "__main__":
    reingest()

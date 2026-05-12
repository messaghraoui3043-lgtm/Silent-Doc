import os
import time
import shutil
import uuid
import fitz
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- CONFIGURATION ----------------
BASE_DIR = os.path.dirname(__file__)
WATCH_DIR = os.path.join(BASE_DIR, "auto_upload")
ARCHIVE_DIR = os.path.join(WATCH_DIR, "archive")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ensure directories exist
os.makedirs(WATCH_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

class MedicalDocumentHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        print(f"[*] Initializing ChromaDB client at {DB_PATH}")
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="medical_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

    def process_file(self, file_path):
        if not (file_path.endswith('.txt') or file_path.endswith('.md') or file_path.endswith('.pdf')):
            return

        filename = os.path.basename(file_path)
        print(f"\n[+] New medical document detected: {filename}", flush=True)
        
        # Add a small delay to ensure file is completely written before reading
        time.sleep(1)

        try:
            content = ""
            if file_path.endswith('.pdf'):
                doc = fitz.open(file_path)
                for page in doc:
                    extracted = page.get_text()
                    if extracted:
                        content += extracted + "\n"
                doc.close()
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

            if not content.strip():
                print(f"[-] File {filename} is empty, skipping.", flush=True)
                return

            print(f"[*] Splitting text into chunks...", flush=True)
            chunks = self.text_splitter.split_text(content)
            
            if not chunks:
                return

            # Generate unique IDs to prevent overwriting
            ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
            metadatas = [{"source": filename} for _ in range(len(chunks))]

            print(f"[*] Embedding and ingesting {len(chunks)} chunks into ChromaDB...", flush=True)
            self.collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
            print(f"[+] Successfully ingested {filename} into RAG database.", flush=True)

            # Move to archive
            dest_path = os.path.join(ARCHIVE_DIR, filename)
            # Handle potential filename collisions in archive
            if os.path.exists(dest_path):
                name, ext = os.path.splitext(filename)
                dest_path = os.path.join(ARCHIVE_DIR, f"{name}_{int(time.time())}{ext}")

            shutil.move(file_path, dest_path)
            print(f"[*] Moved {filename} to archive.", flush=True)

        except Exception as e:
            print(f"[!] Error processing {filename}: {e}", flush=True)

    def on_created(self, event):
        if not event.is_directory:
            self.process_file(event.src_path)
            
    def on_modified(self, event):
        if not event.is_directory:
            self.process_file(event.src_path)
            
    def on_moved(self, event):
        # Handle cases where files are moved/pasted into the directory instead of created
        if not event.is_directory and os.path.dirname(event.dest_path) == WATCH_DIR:
            self.process_file(event.dest_path)

def start_watcher():
    print(f"==================================================", flush=True)
    print(f" Silent Doctor - ChromaDB Auto-Ingestion Watcher ", flush=True)
    print(f"==================================================", flush=True)
    print(f"[*] Watching directory: {WATCH_DIR}", flush=True)
    print(f"[*] Waiting for new .md, .txt, or .pdf files...", flush=True)
    
    event_handler = MedicalDocumentHandler()
    
    # Process existing files before starting the watcher
    print("[*] Checking for existing files in directory...", flush=True)
    for existing_file in os.listdir(WATCH_DIR):
        file_path = os.path.join(WATCH_DIR, existing_file)
        if os.path.isfile(file_path):
            event_handler.process_file(file_path)

    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[*] Stopping watcher...")
    observer.join()

if __name__ == "__main__":
    start_watcher()

import os
import glob
import time
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- CONFIGURATION ----------------
DIR_PATH = os.path.join(os.path.dirname(__file__), "medical_database")
DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 5000 # ChromaDB handles larger batches well

def load_documents():
    """Load all text and markdown files from the target directory."""
    print("[1] Scanning directory for medical documents (.md, .txt)...")
    search_patterns = [os.path.join(DIR_PATH, "*.md"), os.path.join(DIR_PATH, "*.txt")]
    files = []
    for pattern in search_patterns:
        files.extend(glob.glob(pattern))

    print(f"Found {len(files)} files.")
    docs = []
    from io import open
    for file in files:
        try:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if content:
                    docs.append(content)
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            
    return docs

def split_documents(docs):
    """Split medical documents into precise chunks."""
    print(f"[2] Splitting texts with Chunk={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    all_chunks = []
    for doc in docs:
        all_chunks.extend(text_splitter.split_text(doc))
        
    print(f"Successfully generated {len(all_chunks)} text chunks.")
    return all_chunks

def build_chromadb():
    print("--- Phase 3 ChromaDB Ingestion Started ---")
    start_time = time.time()
    
    docs = load_documents()
    if not docs:
        print("No documents found to process. Exiting.")
        return

    text_chunks = split_documents(docs)
    total_chunks = len(text_chunks)
    
    if total_chunks == 0:
        print("Chunking resulted in 0 items. Exiting.")
        return

    print(f"[3] Initializing ChromaDB Persistent Client at {DB_PATH}...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # ChromaDB's default embedding function uses all-MiniLM-L6-v2 which matches the old FAISS model
    collection = client.get_or_create_collection(
        name="medical_knowledge",
        metadata={"hnsw:space": "cosine"}
    )

    print("[4] Batch embedding & indexing all chunks into ChromaDB...")
    try:
        for i in range(0, total_chunks, BATCH_SIZE):
            batch_chunks = text_chunks[i:i + BATCH_SIZE]
            batch_ids = [str(j) for j in range(i, i + len(batch_chunks))]
            print(f"   Processing batch {i} to {i+len(batch_chunks)} / {total_chunks}...")
            
            collection.add(
                documents=batch_chunks,
                ids=batch_ids
            )
            
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return

    print("--------------------------------------------------")
    print(f"INGESTION SUCCESS: Validated {collection.count()} vectors populated in ChromaDB.")
    print(f"Database saved to: {DB_PATH}")
    print(f"Time Taken: {round(time.time() - start_time, 2)} seconds")
    print("--------------------------------------------------")

if __name__ == "__main__":
    build_chromadb()

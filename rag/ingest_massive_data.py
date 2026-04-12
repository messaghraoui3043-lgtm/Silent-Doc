import os
import glob
import pickle
import faiss
import numpy as np
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ---------------- CONFIGURATION ----------------
DIR_PATH = os.path.join(os.path.dirname(__file__), "medical_database")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "moroccan_medical_brain.index")
META_PATH = os.path.join(os.path.dirname(__file__), "moroccan_medical_meta.pkl")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
FAISS_NLIST = 1024
TRAIN_LIMIT = 25000
BATCH_SIZE = 2048

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

def build_faiss_index():
    print("--- Phase 3 FAISS Ingestion Started ---")
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

    print(f"[3] Loading embedding model: {EMBED_MODEL_NAME}...")
    try:
        model = SentenceTransformer(EMBED_MODEL_NAME)
    except Exception as e:
        print(f"Failed to load sentence-transformer: {e}")
        return

    # Prepare FAISS IVF Index
    print(f"[4] Initializing FAISS IndexIVFFlat (Dim: {EMBED_DIM}, Clusters: {FAISS_NLIST})...")
    quantizer = faiss.IndexFlatL2(EMBED_DIM)
    index = faiss.IndexIVFFlat(quantizer, EMBED_DIM, FAISS_NLIST)

    # We must train the IndexIVFFlat
    print(f"[5] Training FAISS with the first Min({TRAIN_LIMIT}, {total_chunks}) vectors...")
    chunks_for_training = text_chunks[:TRAIN_LIMIT]
    try:
        train_embeddings = model.encode(chunks_for_training, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
        # Normalize for inner product / L2 speed stability
        faiss.normalize_L2(train_embeddings)
        index.train(train_embeddings)
        print("Training complete!")
    except Exception as e:
        print(f"Memory/Training Error: {e}")
        return

    # Batch Add Loop
    print("[6] Batch embedding & indexing all chunks...")
    try:
        for i in range(0, total_chunks, BATCH_SIZE):
            batch_chunks = text_chunks[i:i + BATCH_SIZE]
            print(f"   Processing batch {i} to {i+len(batch_chunks)} / {total_chunks}...")
            
            batch_embeddings = model.encode(batch_chunks, batch_size=256, convert_to_numpy=True)
            faiss.normalize_L2(batch_embeddings)
            
            index.add(batch_embeddings)
            
    except MemoryError:
        print("CRITICAL: Ran out of memory during batch execution!")
        return
    except Exception as e:
        print(f"Error during ingestion: {e}")
        return

    # Persistence
    print("[7] Saving Database & Metadata to disk...")
    # Save the index natively
    faiss.write_index(index, INDEX_PATH)
    
    # Save the raw text chunks to map index IDs natively
    with open(META_PATH, "wb") as f:
        pickle.dump(text_chunks, f)
        
    print("--------------------------------------------------")
    print(f"INGESTION SUCCESS: Validated {index.ntotal} vectors populated.")
    print(f"Index saved to: {INDEX_PATH}")
    print(f"Meta saved to: {META_PATH}")
    print(f"Time Taken: {round(time.time() - start_time, 2)} seconds")
    print("--------------------------------------------------")

if __name__ == "__main__":
    build_faiss_index()

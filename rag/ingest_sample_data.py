import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

def ingest_data():
    root_dir = Path(__file__).resolve().parent.parent
    data_path = root_dir / "rag" / "sample_medical_data.txt"
    index_dir = root_dir / "rag" / "index"
    index_path = index_dir / "medical_index.faiss"

    print("[Ingest] Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    if not data_path.exists():
        print(f"[Error] File not found: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into entries separated by empty lines
    chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
    print(f"[Ingest] Found {len(chunks)} text chunks.")

    print("[Ingest] Encoding text chunks...")
    embeddings = model.encode(chunks, show_progress_bar=False)
    dimension = embeddings.shape[1]

    print("[Ingest] Building FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, str(index_path))
    
    # Save the raw text chunks to map back results
    chunks_path = index_dir / "medical_chunks.txt"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            # use a distinct separator
            f.write(chunk.replace('\n', '|||') + "\n")

    print(f"[Ingest] Successfully saved index to {index_path} and chunks map to {chunks_path}.")

if __name__ == "__main__":
    ingest_data()

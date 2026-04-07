import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# Disable progress bars properly for Windows background services
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def extract_text_from_file(file_path):
    ext = file_path.suffix.lower()
    text = ""
    
    if ext == ".pdf":
        try:
            reader = PdfReader(str(file_path))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"[Ingest] Error reading PDF {file_path.name}: {e}")
            
    elif ext in [".txt", ".md"]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"[Ingest] Error reading text file {file_path.name}: {e}")
            
    return text

def ingest_massive_data():
    root_dir = Path(__file__).resolve().parent.parent
    db_dir = root_dir / "rag" / "medical_database"
    index_dir = root_dir / "rag" / "index"
    index_path = index_dir / "medical_index.faiss"
    chunks_path = index_dir / "medical_chunks.txt"

    if not db_dir.exists():
        print(f"[Error] Database directory not found at {db_dir}")
        return

    print("[Ingest] Scanning medical database directory...")
    all_text = ""
    file_count = 0
    for file_path in db_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in [".pdf", ".txt", ".md"]:
            print(f"  -> Reading {file_path.name}...")
            all_text += extract_text_from_file(file_path) + "\n\n"
            file_count += 1

    if file_count == 0 or not all_text.strip():
        print("[Ingest] No text data found to process. Dropping some PDFs in rag/medical_database/!")
        return

    print(f"[Ingest] Finished reading {file_count} files. Preparing LangChain text splitter...")
    
    # Langchain splitting Strategy
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_text(all_text)
    print(f"[Ingest] Created {len(chunks)} contextual chunks.")

    print("[Ingest] Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    print("[Ingest] Encoding massive text chunks into vectors...")
    # Safe progress bar override
    embeddings = model.encode(chunks, show_progress_bar=False)
    dimension = embeddings.shape[1]

    print("[Ingest] Building FAISS Index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, str(index_path))
    
    # Save the raw text chunks to map back results
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            # use a distinct separator
            f.write(chunk.replace('\n', '|||') + "\n")

    print(f"[Ingest] Successfully saved index with {len(chunks)} vectors to {index_path}!")

if __name__ == "__main__":
    ingest_massive_data()

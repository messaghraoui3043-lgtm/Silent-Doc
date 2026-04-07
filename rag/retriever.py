import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

class MedicalRetriever:
    def __init__(self, 
                 index_path="rag/index/medical_index.faiss",
                 chunks_path="rag/index/medical_chunks.txt",
                 model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        
        self.root_dir = Path(__file__).resolve().parent.parent
        self.index_path = self.root_dir / index_path
        self.chunks_path = self.root_dir / chunks_path
        
        print(f"[RAG] Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        self.chunks_data = []
        self.index = None
        
        self._load_chunks()
        self._load_index()

    def _load_chunks(self):
        if not self.chunks_path.exists():
            print(f"[RAG] Warning: Knowledge chunks not found at {self.chunks_path}. Run ingest script first.")
            return
            
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.chunks_data = [line.strip().replace('|||', '\n') for line in lines]
            
    def _load_index(self):
        if not self.index_path.exists():
            print(f"[RAG] Warning: Index not found at {self.index_path}. Run ingest script first.")
            return
            
        print(f"[RAG] Loading existing FAISS index from {self.index_path}...")
        self.index = faiss.read_index(self.index_path.as_posix())

    def get_context(self, query, top_k=2):
        """
        Retrieves the most relevant medical entries for a given query.
        """
        if self.index is None or not self.chunks_data:
            return None
            
        query_embedding = self.model.encode([query], show_progress_bar=False)
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = []
        # Return top N matches
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks_data):
                results.append(self.chunks_data[idx])
        
        if not results:
            return None
            
        return "\n\n".join(results)

# Singleton instance for the API
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = MedicalRetriever()
    return _retriever

if __name__ == "__main__":
    # Test
    r = get_retriever()
    ctx = r.get_context("راسي كايضرني")
    print("\n[TEST] Retrieval for 'راسي كايضرني':")
    print(ctx)

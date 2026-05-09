import os
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

class MedicalRetriever:
    def __init__(self, db_path="rag/chroma_db"):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.db_path = self.root_dir / db_path
        
        self.client = None
        self.collection = None
        
        self._init_db()

    def _init_db(self):
        if not self.db_path.exists():
            print(f"[RAG] Warning: ChromaDB not found at {self.db_path}. Run ingest script first.")
            return
            
        print(f"[RAG] Connecting to ChromaDB at {self.db_path}...")
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        try:
            self.collection = self.client.get_collection(name="medical_knowledge")
        except Exception as e:
            print(f"[RAG] Warning: Could not load collection 'medical_knowledge': {e}")

    def get_context(self, query, top_k=2):
        """
        Retrieves the most relevant medical entries for a given query.
        """
        if self.collection is None:
            return None
            
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        documents = results.get("documents", [])
        if not documents or not documents[0]:
            return None
            
        return "\n\n".join(documents[0])

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

class LinguisticRetriever:
    def __init__(self, db_path="rag/chroma_db"):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.db_path = self.root_dir / db_path
        
        self.client = None
        self.collection = None
        
        self._init_db()

    def _init_db(self):
        if not self.db_path.exists():
            print(f"[RAG] Warning: ChromaDB not found at {self.db_path}. Run ingest script first.")
            return
            
        print(f"[RAG] Connecting to ChromaDB at {self.db_path} for Linguistic Retrieval...")
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        try:
            self.collection = self.client.get_collection(name="linguistic_knowledge")
        except Exception as e:
            print(f"[RAG] Warning: Could not load collection 'linguistic_knowledge': {e}")

    def get_linguistic_context(self, query, top_k=2):
        if self.collection is None:
            return None
            
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # We need to return the Doctor Responses, which are stored in the metadatas array
        metadatas = results.get("metadatas", [])
        if not metadatas or not metadatas[0]:
            return None
            
        responses = [meta["response"] for meta in metadatas[0] if "response" in meta]
        
        if not responses:
            return None
            
        return "\n\n".join(responses)

_ling_retriever = None

def get_linguistic_retriever():
    global _ling_retriever
    if _ling_retriever is None:
        _ling_retriever = LinguisticRetriever()
    return _ling_retriever

"""
Silent Doctor — Vector Store Module
=====================================
FAISS-based vector store for medical knowledge retrieval.

Supports building, saving, loading, and searching a FAISS index.
Includes a document ingestion pipeline for PDFs and text files.

Usage:
    store = VectorStore()
    store.build_from_directory("datasets/")
    store.save()

    results = store.search("How to treat eczema?", k=5)
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATASETS_DIR,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    RAG_TOP_K,
)
from rag.embeddings import get_embeddings
from utils.helpers import setup_logger

logger = setup_logger(__name__)


class VectorStore:
    """
    FAISS-based vector store for medical document retrieval.

    Stores document chunks with their embeddings for fast
    similarity search. Supports persistence (save/load).
    """

    def __init__(
        self,
        index_path: Optional[str | Path] = None,
        metadata_path: Optional[str | Path] = None,
    ):
        self.index_path = Path(index_path) if index_path else FAISS_INDEX_PATH
        self.metadata_path = (
            Path(metadata_path) if metadata_path else FAISS_METADATA_PATH
        )

        self._index = None
        self._documents: list[dict] = []  # {text, source, chunk_id}
        self._embedder = get_embeddings()

        # Try to load existing index
        if self.index_path.exists() and self.metadata_path.exists():
            self.load()

    @property
    def index(self):
        """Access the FAISS index."""
        return self._index

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal if self._index else 0

    # ── Document Ingestion ──────────────────────────────────────────────

    def _chunk_text(self, text: str, source: str = "") -> list[dict]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "text": chunk,
                    "source": source,
                    "chunk_id": chunk_id,
                })
                chunk_id += 1

            start += CHUNK_SIZE - CHUNK_OVERLAP

        return chunks

    def _read_pdf(self, pdf_path: str | Path) -> str:
        """Extract text from a PDF file."""
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(pdf_path))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n".join(text_parts)
        except ImportError:
            logger.warning("PyPDF2 not installed. Skipping PDF: {pdf_path}")
            return ""
        except Exception as exc:
            logger.error(f"Error reading PDF {pdf_path}: {exc}")
            return ""

    def _read_text_file(self, file_path: str | Path) -> str:
        """Read a plain text file."""
        try:
            return Path(file_path).read_text(encoding="utf-8")
        except Exception as exc:
            logger.error(f"Error reading {file_path}: {exc}")
            return ""

    def ingest_file(self, file_path: str | Path) -> int:
        """
        Ingest a single file (PDF or text) into the store.

        Returns:
            Number of chunks created.
        """
        path = Path(file_path)
        logger.info(f"📄 Ingesting: {path.name}")

        if path.suffix.lower() == ".pdf":
            text = self._read_pdf(path)
        elif path.suffix.lower() in (".txt", ".md", ".csv"):
            text = self._read_text_file(path)
        else:
            logger.warning(f"Unsupported file type: {path.suffix}")
            return 0

        if not text.strip():
            logger.warning(f"No text extracted from {path.name}")
            return 0

        chunks = self._chunk_text(text, source=str(path.name))
        self._documents.extend(chunks)

        logger.info(f"  → Created {len(chunks)} chunks from {path.name}")
        return len(chunks)

    def build_from_directory(
        self,
        directory: str | Path = DATASETS_DIR,
        extensions: tuple = (".pdf", ".txt", ".md"),
    ) -> int:
        """
        Ingest all supported files from a directory and build the index.

        Returns:
            Total number of chunks indexed.
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return 0

        files = [
            f for f in dir_path.rglob("*")
            if f.suffix.lower() in extensions
        ]

        if not files:
            logger.warning(f"No supported files found in {directory}")
            return 0

        logger.info(f"📚 Found {len(files)} files to ingest.")

        total_chunks = 0
        for file_path in files:
            total_chunks += self.ingest_file(file_path)

        # Build FAISS index from all documents
        self._build_index()

        logger.info(
            f"✅ Index built: {total_chunks} chunks, "
            f"{self.size} vectors"
        )
        return total_chunks

    def _build_index(self):
        """Build the FAISS index from current documents."""
        import faiss

        if not self._documents:
            logger.warning("No documents to index.")
            return

        texts = [doc["text"] for doc in self._documents]
        embeddings = self._embedder.embed_documents(texts)
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # Create a flat (exact search) index
        dimension = embeddings_np.shape[1]
        self._index = faiss.IndexFlatL2(dimension)
        self._index.add(embeddings_np)

        logger.info(
            f"FAISS index created: {self._index.ntotal} vectors, "
            f"dimension={dimension}"
        )

    # ── Search ──────────────────────────────────────────────────────────

    def search(self, query: str, k: int = RAG_TOP_K) -> list[dict]:
        """
        Search for the most relevant document chunks.

        Args:
            query: Search query text.
            k: Number of results to return.

        Returns:
            List of dicts with keys: text, source, score, chunk_id
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Index is empty. No results returned.")
            return []

        # Embed the query
        query_embedding = np.array(
            [self._embedder.embed_query(query)],
            dtype=np.float32,
        )

        # Search
        distances, indices = self._index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self._documents):
                doc = self._documents[idx]
                results.append({
                    "text": doc["text"],
                    "source": doc["source"],
                    "chunk_id": doc["chunk_id"],
                    "score": float(dist),
                })

        logger.info(f"🔍 Found {len(results)} results for query.")
        return results

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self):
        """Save the FAISS index and metadata to disk."""
        import faiss

        if self._index is None:
            logger.warning("No index to save.")
            return

        # Ensure directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self._documents, f)

        logger.info(
            f"💾 Index saved: {self.index_path} "
            f"({self.size} vectors)"
        )

    def load(self):
        """Load a FAISS index and metadata from disk."""
        import faiss

        if not self.index_path.exists():
            logger.warning(f"Index file not found: {self.index_path}")
            return

        self._index = faiss.read_index(str(self.index_path))

        if self.metadata_path.exists():
            with open(self.metadata_path, "rb") as f:
                self._documents = pickle.load(f)

        logger.info(
            f"✅ Index loaded: {self.size} vectors, "
            f"{len(self._documents)} documents"
        )


# ── Convenience function ────────────────────────────────────────────────

_cached_store: Optional[VectorStore] = None


def get_vector_store(**kwargs) -> VectorStore:
    """Get or create a cached VectorStore instance."""
    global _cached_store
    if _cached_store is None:
        _cached_store = VectorStore(**kwargs)
    return _cached_store

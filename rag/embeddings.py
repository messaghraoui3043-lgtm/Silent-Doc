"""
Silent Doctor — Embedding Module
==================================
Local text embeddings using sentence-transformers (all-MiniLM-L6-v2).

Implements the LangChain Embeddings interface for seamless integration.

Usage:
    embedder = LocalEmbeddings()
    vector = embedder.embed_query("What causes eczema?")
    vectors = embedder.embed_documents(["doc1", "doc2", "doc3"])
"""

from typing import Optional

from config.settings import EMBEDDING_MODEL
from utils.helpers import setup_logger

logger = setup_logger(__name__)


class LocalEmbeddings:
    """
    Local embedding model using sentence-transformers.

    Model: all-MiniLM-L6-v2
        - 384 dimensions
        - ~80 MB
        - Fast inference on CPU

    Implements embed_query() and embed_documents() to be compatible
    with LangChain's Embeddings interface.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            logger.info(
                f"✅ Embedding model loaded. "
                f"Dimension: {self._model.get_sentence_embedding_dimension()}"
            )
        return self._model

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.

        Args:
            text: The query string.

        Returns:
            List of floats (embedding vector).
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of document texts.

        Args:
            texts: List of document strings.

        Returns:
            List of embedding vectors.
        """
        logger.info(f"Embedding {len(texts)} documents ...")
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32,
        )
        logger.info("✅ Batch embedding complete.")
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self.model.get_sentence_embedding_dimension()


# ── Convenience function ────────────────────────────────────────────────

_cached_embeddings: Optional[LocalEmbeddings] = None


def get_embeddings(**kwargs) -> LocalEmbeddings:
    """Get or create a cached LocalEmbeddings instance."""
    global _cached_embeddings
    if _cached_embeddings is None:
        _cached_embeddings = LocalEmbeddings(**kwargs)
    return _cached_embeddings

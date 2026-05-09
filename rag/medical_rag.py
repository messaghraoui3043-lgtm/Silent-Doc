"""
Silent Doctor — Medical RAG Pipeline
======================================
Retrieval-Augmented Generation for medical question answering.

Combines FAISS vector search with Ollama LLM (Gemma 2B)
for context-grounded medical responses.

Usage:
    rag = MedicalRAG()
    answer = rag.ask("What are the symptoms of eczema?")
"""

from typing import Optional

from config.settings import (
    MEDICAL_SYSTEM_PROMPT,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RAG_TOP_K,
)
from rag.vector_store import get_vector_store
from utils.helpers import setup_logger

logger = setup_logger(__name__)


class MedicalRAG:
    """
    Medical RAG pipeline.

    Flow:
        1. User asks a question
        2. Question is embedded and searched against FAISS index
        3. Top-k relevant document chunks are retrieved
        4. Context + question are sent to Ollama LLM
        5. LLM generates a grounded, cautious medical response
    """

    def __init__(
        self,
        ollama_model: str = OLLAMA_MODEL,
        ollama_base_url: str = OLLAMA_BASE_URL,
        system_prompt: str = MEDICAL_SYSTEM_PROMPT,
        top_k: int = RAG_TOP_K,
    ):
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.system_prompt = system_prompt
        self.top_k = top_k

        self._vector_store = get_vector_store()
        self._llm = None

    @property
    def llm(self):
        """Lazy-load the LangChain Ollama LLM."""
        if self._llm is None:
            logger.info(
                f"Connecting to Ollama: {self.ollama_model} "
                f"@ {self.ollama_base_url}"
            )
            try:
                from langchain_community.llms import Ollama

                self._llm = Ollama(
                    model=self.ollama_model,
                    base_url=self.ollama_base_url,
                    temperature=0.3,  # Lower temperature for medical accuracy
                )
                logger.info("✅ Ollama LLM connected.")
            except ImportError:
                logger.error(
                    "langchain-community not installed. "
                    "Install with: pip install langchain-community"
                )
                raise
        return self._llm

    def _retrieve_context(self, question: str) -> str:
        """
        Retrieve relevant medical context from the vector store.

        Args:
            question: User's medical question.

        Returns:
            Formatted context string from top-k documents.
        """
        results = self._vector_store.search(question, k=self.top_k)

        if not results:
            logger.info("No relevant context found in knowledge base.")
            return "No specific medical reference available."

        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {doc['source']}]\n{doc['text']}"
            )

        context = "\n\n".join(context_parts)
        logger.info(
            f"📚 Retrieved {len(results)} context chunks "
            f"({len(context)} chars)"
        )
        return context

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build the full prompt with system instructions, context, and question.
        """
        return f"""{self.system_prompt}

--- MEDICAL CONTEXT ---
{context}
--- END CONTEXT ---

PATIENT QUESTION: {question}

Provide a helpful, empathetic response based on the medical context above.
Always remind the patient that this is AI-generated advice and they should
consult a healthcare professional for proper diagnosis and treatment.

RESPONSE:"""

    def ask(self, question: str) -> dict:
        """
        Answer a medical question using RAG + LLM.

        Args:
            question: The user's medical question.

        Returns:
            dict with keys:
                - answer (str): The LLM's response
                - context_sources (list): Sources used for context
                - model (str): The LLM model used
        """
        logger.info(f"❓ Medical query: '{question[:80]}...'")

        # Step 1: Retrieve relevant context
        context = self._retrieve_context(question)

        # Step 2: Build prompt
        prompt = self._build_prompt(question, context)

        # Step 3: Get LLM response
        logger.info("🧠 Generating response with LLM ...")
        answer = self.llm.invoke(prompt)

        # Extract source names
        results = self._vector_store.search(question, k=self.top_k)
        sources = list({doc["source"] for doc in results})

        logger.info(f"✅ Response generated ({len(answer)} chars)")

        return {
            "answer": answer.strip(),
            "context_sources": sources,
            "model": self.ollama_model,
        }

    def ask_without_rag(self, question: str) -> str:
        """
        Ask the LLM directly without RAG context.
        Useful as a fallback when no knowledge base is available.
        """
        prompt = f"""{self.system_prompt}

PATIENT QUESTION: {question}

Provide a helpful, empathetic response.
Always remind the patient to consult a healthcare professional.

RESPONSE:"""

        return self.llm.invoke(prompt).strip()


# ── Convenience function ────────────────────────────────────────────────

_cached_rag: Optional[MedicalRAG] = None


def get_medical_rag(**kwargs) -> MedicalRAG:
    """Get or create a cached MedicalRAG instance."""
    global _cached_rag
    if _cached_rag is None:
        _cached_rag = MedicalRAG(**kwargs)
    return _cached_rag

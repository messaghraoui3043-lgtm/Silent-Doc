"""
Silent Doctor — Child Health Support Module
=============================================
Specialized module for pediatric health assistance.

Uses the same RAG + LLM pipeline with a child-health-focused
prompt template and optional separate FAISS index.

Usage:
    child = ChildSupport()
    result = child.assess("My child has a fever and rash")
"""

from typing import Optional

from config.settings import (
    CHILD_HEALTH_SYSTEM_PROMPT,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RAG_TOP_K,
)
from rag.vector_store import get_vector_store
from utils.helpers import setup_logger

logger = setup_logger(__name__)


# ── Common childhood symptoms for keyword matching ──────────────────────
CHILD_SYMPTOMS = {
    "fever": [
        "fever", "temperature", "hot", "warm", "burning up",
        "حمى", "سخونة",
    ],
    "rash": [
        "rash", "spots", "bumps", "red skin", "hives",
        "طفح", "بقع",
    ],
    "eye_infection": [
        "eye", "red eye", "swollen eye", "discharge", "itchy eye",
        "عين", "احمرار",
    ],
    "skin_irritation": [
        "irritation", "itchy", "dry skin", "peeling", "eczema",
        "حكة", "جفاف",
    ],
}


class ChildSupport:
    """
    Child health assessment module.

    Analyzes symptoms reported by parents and provides age-appropriate
    medical guidance. Uses RAG for evidence-based responses and
    a specialized prompt for pediatric care.
    """

    def __init__(
        self,
        ollama_model: str = OLLAMA_MODEL,
        ollama_base_url: str = OLLAMA_BASE_URL,
        top_k: int = RAG_TOP_K,
    ):
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.top_k = top_k

        self._vector_store = get_vector_store()
        self._llm = None

    @property
    def llm(self):
        """Lazy-load the Ollama LLM."""
        if self._llm is None:
            from langchain_community.llms import Ollama

            self._llm = Ollama(
                model=self.ollama_model,
                base_url=self.ollama_base_url,
                temperature=0.3,
            )
            logger.info("✅ Child support LLM connected.")
        return self._llm

    def detect_symptom_categories(self, description: str) -> list[str]:
        """
        Detect which symptom categories are mentioned in the description.

        Returns:
            List of matched category names (e.g., ["fever", "rash"]).
        """
        description_lower = description.lower()
        matched = []

        for category, keywords in CHILD_SYMPTOMS.items():
            if any(kw in description_lower for kw in keywords):
                matched.append(category)

        return matched

    def _retrieve_context(self, question: str) -> str:
        """Retrieve relevant pediatric context from vector store."""
        results = self._vector_store.search(question, k=self.top_k)

        if not results:
            return "No specific pediatric reference available."

        context_parts = [
            f"[Source {i}: {doc['source']}]\n{doc['text']}"
            for i, doc in enumerate(results, 1)
        ]
        return "\n\n".join(context_parts)

    def assess(
        self,
        symptom_description: str,
        child_age: Optional[str] = None,
    ) -> dict:
        """
        Assess a child's symptoms and provide guidance.

        Args:
            symptom_description: Parent's description of symptoms.
            child_age: Optional age of the child (e.g., "2 years").

        Returns:
            dict with keys:
                - advice (str): Medical guidance
                - detected_categories (list): Matched symptom categories
                - severity_hint (str): low, medium, or high
                - sources (list): Knowledge sources used
        """
        logger.info(f"👶 Child health query: '{symptom_description[:80]}...'")

        # Detect symptom categories
        categories = self.detect_symptom_categories(symptom_description)

        # Build enriched query
        age_info = f"Child age: {child_age}. " if child_age else ""
        enriched_query = (
            f"{age_info}Child symptoms: {symptom_description}. "
            f"Categories: {', '.join(categories) if categories else 'general'}."
        )

        # Retrieve context
        context = self._retrieve_context(enriched_query)

        # Build prompt
        prompt = f"""{CHILD_HEALTH_SYSTEM_PROMPT}

--- MEDICAL CONTEXT ---
{context}
--- END CONTEXT ---

PARENT'S DESCRIPTION: {symptom_description}
{f"CHILD'S AGE: {child_age}" if child_age else ""}
DETECTED SYMPTOM CATEGORIES: {', '.join(categories) if categories else 'Not specifically categorized'}

Provide helpful, reassuring guidance for the parent. Include:
1. What the symptoms might indicate
2. Home care suggestions
3. Warning signs that require immediate medical attention
4. When to see a doctor

RESPONSE:"""

        # Get LLM response
        advice = self.llm.invoke(prompt).strip()

        # Simple severity heuristic
        severity = self._estimate_severity(categories, symptom_description)

        # Extract sources
        results = self._vector_store.search(enriched_query, k=self.top_k)
        sources = list({doc["source"] for doc in results})

        logger.info(f"✅ Child health assessment complete (severity: {severity})")

        return {
            "advice": advice,
            "detected_categories": categories,
            "severity_hint": severity,
            "sources": sources,
        }

    def _estimate_severity(
        self,
        categories: list[str],
        description: str,
    ) -> str:
        """
        Simple heuristic to estimate symptom severity.
        This is NOT a diagnosis — it helps prioritize responses.
        """
        high_risk_keywords = [
            "breathing", "unconscious", "seizure", "blood",
            "severe", "emergency", "not responding", "blue",
        ]
        description_lower = description.lower()

        if any(kw in description_lower for kw in high_risk_keywords):
            return "high"
        if len(categories) >= 3:
            return "medium"
        if len(categories) >= 1:
            return "low"
        return "low"


# ── Convenience function ────────────────────────────────────────────────

_cached_child_support: Optional[ChildSupport] = None


def get_child_support(**kwargs) -> ChildSupport:
    """Get or create a cached ChildSupport instance."""
    global _cached_child_support
    if _cached_child_support is None:
        _cached_child_support = ChildSupport(**kwargs)
    return _cached_child_support

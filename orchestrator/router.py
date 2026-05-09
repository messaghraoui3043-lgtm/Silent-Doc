"""
Silent Doctor — LangChain Orchestrator
========================================
Central routing system that detects input type and dispatches
to the appropriate processing pipeline.

Supported input types:
    - skin_image  → MobileNetV2 skin disease classifier
    - eye_image   → MobileNetV3 eye disease classifier
    - voice       → STT → Translation → LLM → Translation → TTS
    - question    → RAG + Ollama LLM
    - child       → Child health RAG + LLM

Uses LangChain RunnableLambda for clean, composable routing.

Usage:
    orchestrator = Orchestrator()
    result = orchestrator.route({
        "input_type": "question",
        "data": "What causes eczema?"
    })
"""

from typing import Any, Optional

from langchain_core.runnables import RunnableLambda

from config.settings import LANGUAGE_CODES
from utils.helpers import setup_logger

logger = setup_logger(__name__)


class Orchestrator:
    """
    Central orchestrator that routes requests to the correct AI pipeline.

    Each pipeline is a LangChain RunnableLambda for composability.
    Models are loaded lazily — nothing is initialized until first use.
    """

    VALID_INPUT_TYPES = {
        "skin_image",
        "eye_image",
        "voice",
        "question",
        "child",
    }

    def __init__(self):
        # Lazy-loaded pipeline references
        self._skin_pipeline = None
        self._eye_pipeline = None
        self._voice_pipeline = None
        self._question_pipeline = None
        self._child_pipeline = None

        logger.info("🏥 Silent Doctor Orchestrator initialized.")

    # ── Pipeline Factories (lazy) ───────────────────────────────────────

    @property
    def skin_pipeline(self) -> RunnableLambda:
        """Skin disease detection pipeline."""
        if self._skin_pipeline is None:
            self._skin_pipeline = RunnableLambda(self._handle_skin_image)
        return self._skin_pipeline

    @property
    def eye_pipeline(self) -> RunnableLambda:
        """Eye disease detection pipeline."""
        if self._eye_pipeline is None:
            self._eye_pipeline = RunnableLambda(self._handle_eye_image)
        return self._eye_pipeline

    @property
    def voice_pipeline(self) -> RunnableLambda:
        """Full voice consultation pipeline."""
        if self._voice_pipeline is None:
            self._voice_pipeline = RunnableLambda(self._handle_voice)
        return self._voice_pipeline

    @property
    def question_pipeline(self) -> RunnableLambda:
        """Text question → RAG → LLM pipeline."""
        if self._question_pipeline is None:
            self._question_pipeline = RunnableLambda(self._handle_question)
        return self._question_pipeline

    @property
    def child_pipeline(self) -> RunnableLambda:
        """Child health assessment pipeline."""
        if self._child_pipeline is None:
            self._child_pipeline = RunnableLambda(self._handle_child)
        return self._child_pipeline

    # ── Routing ─────────────────────────────────────────────────────────

    def route(self, request: dict) -> dict:
        """
        Route a request to the appropriate pipeline.

        Args:
            request: dict with keys:
                - input_type (str): One of VALID_INPUT_TYPES
                - data: The input data (image path, audio path, text, etc.)
                - Additional keys depend on input_type.

        Returns:
            dict with pipeline-specific results.

        Raises:
            ValueError: If input_type is invalid.
        """
        input_type = request.get("input_type", "").lower()

        if input_type not in self.VALID_INPUT_TYPES:
            raise ValueError(
                f"Invalid input_type: '{input_type}'. "
                f"Must be one of: {self.VALID_INPUT_TYPES}"
            )

        logger.info(f"🔀 Routing request: input_type={input_type}")

        # Route to the appropriate pipeline
        pipeline_map = {
            "skin_image": self.skin_pipeline,
            "eye_image": self.eye_pipeline,
            "voice": self.voice_pipeline,
            "question": self.question_pipeline,
            "child": self.child_pipeline,
        }

        pipeline = pipeline_map[input_type]
        result = pipeline.invoke(request)

        logger.info(f"✅ Request processed: input_type={input_type}")
        return result

    # ── Pipeline Handlers ───────────────────────────────────────────────

    def _handle_skin_image(self, request: dict) -> dict:
        """Process a skin disease detection request."""
        from models.skin_model import get_skin_model

        image_path = request["data"]
        logger.info(f"🔬 Analyzing skin image: {image_path}")

        model = get_skin_model()
        prediction = model.predict(image_path)

        # Enrich with medical advice via LLM
        advice = self._get_condition_advice(
            condition=prediction["prediction"],
            body_part="skin",
            confidence=prediction["confidence"],
        )

        return {
            "type": "skin_analysis",
            "prediction": prediction,
            "advice": advice,
        }

    def _handle_eye_image(self, request: dict) -> dict:
        """Process an eye disease detection request."""
        from models.eye_model import get_eye_model

        image_path = request["data"]
        logger.info(f"👁️ Analyzing eye image: {image_path}")

        model = get_eye_model()
        prediction = model.predict(image_path)

        # Enrich with medical advice via LLM
        advice = self._get_condition_advice(
            condition=prediction["prediction"],
            body_part="eye",
            confidence=prediction["confidence"],
        )

        return {
            "type": "eye_analysis",
            "prediction": prediction,
            "advice": advice,
        }

    def _handle_voice(self, request: dict) -> dict:
        """
        Full voice consultation pipeline:
            1. STT (Whisper)
            2. Translate to English (NLLB)
            3. Query LLM (RAG + Ollama)
            4. Translate response back
            5. TTS (Coqui)
        """
        from voice.speech_to_text import get_stt
        from voice.translation import get_translator
        from voice.tts import get_tts
        from rag.medical_rag import get_medical_rag

        audio_input = request.get("data")
        target_language = request.get("language", "darija")

        # Step 1: Speech-to-Text
        logger.info("Step 1/5: Speech-to-Text ...")
        stt = get_stt()

        if audio_input is None:
            # Record from microphone
            duration = request.get("duration", 5.0)
            stt_result = stt.transcribe_from_mic(duration=duration)
        else:
            stt_result = stt.transcribe(audio_input)

        original_text = stt_result["text"]
        detected_language = stt_result["language"]
        logger.info(f"  Transcribed ({detected_language}): {original_text[:80]}")

        # Step 2: Translate to English
        logger.info("Step 2/5: Translation → English ...")
        translator = get_translator()

        if detected_language in ("en", "eng"):
            english_text = original_text
        else:
            # Map Whisper language codes to our language names
            lang_map = {"ar": "arabic", "fr": "french"}
            src_lang = lang_map.get(detected_language, target_language)
            english_text = translator.translate_to_english(
                original_text, src_lang=src_lang
            )
        logger.info(f"  English: {english_text[:80]}")

        # Step 3: Query medical LLM with RAG
        logger.info("Step 3/5: RAG + LLM ...")
        rag = get_medical_rag()
        rag_result = rag.ask(english_text)
        english_answer = rag_result["answer"]
        logger.info(f"  LLM answer: {english_answer[:80]}")

        # Step 4: Translate response back
        logger.info(f"Step 4/5: Translation → {target_language} ...")
        if target_language in ("english", "en"):
            translated_answer = english_answer
        else:
            translated_answer = translator.translate_from_english(
                english_answer, tgt_lang=target_language
            )
        logger.info(f"  Translated: {translated_answer[:80]}")

        # Step 5: Text-to-Speech
        logger.info("Step 5/5: Text-to-Speech ...")
        tts = get_tts()
        tts.speak(translated_answer, language=target_language)

        return {
            "type": "voice_consultation",
            "original_text": original_text,
            "detected_language": detected_language,
            "english_query": english_text,
            "english_answer": english_answer,
            "translated_answer": translated_answer,
            "target_language": target_language,
            "sources": rag_result.get("context_sources", []),
        }

    def _handle_question(self, request: dict) -> dict:
        """Process a text-based medical question via RAG + LLM."""
        from rag.medical_rag import get_medical_rag

        question = request["data"]
        logger.info(f"❓ Text query: {question[:80]}")

        rag = get_medical_rag()
        result = rag.ask(question)

        return {
            "type": "text_answer",
            "question": question,
            "answer": result["answer"],
            "sources": result["context_sources"],
            "model": result["model"],
        }

    def _handle_child(self, request: dict) -> dict:
        """Process a child health assessment request."""
        from modules.child_support import get_child_support

        description = request["data"]
        child_age = request.get("child_age")
        logger.info(f"👶 Child health query: {description[:80]}")

        child = get_child_support()
        result = child.assess(description, child_age=child_age)

        return {
            "type": "child_assessment",
            "description": description,
            "advice": result["advice"],
            "categories": result["detected_categories"],
            "severity": result["severity_hint"],
            "sources": result["sources"],
        }

    # ── Helper Methods ──────────────────────────────────────────────────

    def _get_condition_advice(
        self,
        condition: str,
        body_part: str,
        confidence: float,
    ) -> str:
        """
        Get medical advice for a detected condition using RAG + LLM.

        Returns advice string, or a default message if LLM is unavailable.
        """
        try:
            from rag.medical_rag import get_medical_rag

            rag = get_medical_rag()
            question = (
                f"A {body_part} analysis detected '{condition}' "
                f"with {confidence:.0%} confidence. "
                f"What should the patient know about this condition? "
                f"What are recommended next steps?"
            )
            result = rag.ask(question)
            return result["answer"]
        except Exception as exc:
            logger.warning(f"Could not get LLM advice: {exc}")
            return (
                f"Detected {body_part} condition: {condition} "
                f"(confidence: {confidence:.0%}). "
                f"Please consult a healthcare professional for proper evaluation."
            )

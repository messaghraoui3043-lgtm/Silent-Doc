"""
Silent Doctor — Speech-to-Text Module
=======================================
Local speech recognition using faster-whisper (CTranslate2-based Whisper).

Fully offline — no API calls required.

Usage:
    stt = SpeechToText()
    result = stt.transcribe("audio.wav")
    print(result["text"], result["language"])
"""

import sys
import os
from pathlib import Path
from typing import Optional, Union

# Add parent directory to sys.path for direct execution testing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


from config.settings import (
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
)
from utils.helpers import setup_logger

logger = setup_logger(__name__)


class SpeechToText:
    """
    Local speech-to-text using openai-whisper.
    """

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    @property
    def model(self):
        """Lazy-load the Whisper model on first use."""
        if self._model is None:
            logger.info(
                f"Loading Whisper model: {self.model_size} "
                f"(device={self.device})"
            )
            import whisper

            self._model = whisper.load_model(
                self.model_size,
                device=self.device,
            )
            logger.info("✅ Whisper model loaded.")
        return self._model

    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
    ) -> dict:
        """
        Transcribe audio to text.

        Args:
            audio_input: Path to audio file, or numpy float32 array.
            language: Optional language hint (e.g., "ar" for Arabic).
                      If None, Whisper auto-detects the language.

        Returns:
            dict with keys:
                - text (str): Full transcribed text
                - language (str): Detected language code
                - segments (list): Individual segments with timestamps
        """
        # Handle file path vs numpy array
        if isinstance(audio_input, (str, Path)):
            audio_path = str(audio_input)
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            source = audio_path
        else:
            source = audio_input

        logger.info("🎧 Transcribing audio ...")

        result = self.model.transcribe(
            source,
            language=language,
        )

        # Collect all segments
        segments = []
        for segment in result["segments"]:
            segments.append({
                "start": round(segment["start"], 2),
                "end": round(segment["end"], 2),
                "text": segment["text"].strip(),
            })

        full_text = result["text"].strip()
        detected_lang = result.get("language", language or "unknown")

        logger.info(
            f"✅ Transcription complete. "
            f"Language: {detected_lang}, Length: {len(full_text)} chars"
        )

        return {
            "text": full_text,
            "language": detected_lang,
            "segments": segments,
        }

    def transcribe_from_mic(self, duration: float = 5.0) -> dict:
        """
        Record from microphone and transcribe.

        Args:
            duration: Recording duration in seconds.

        Returns:
            Same dict as transcribe().
        """
        from utils.helpers import record_audio

        audio = record_audio(duration_seconds=duration)
        return self.transcribe(audio)


# ── Convenience function ────────────────────────────────────────────────

_cached_stt: Optional[SpeechToText] = None


def get_stt(**kwargs) -> SpeechToText:
    """Get or create a cached SpeechToText instance."""
    global _cached_stt
    if _cached_stt is None:
        _cached_stt = SpeechToText(**kwargs)
    return _cached_stt


if __name__ == "__main__":
    stt = get_stt()
    print("Testing Whisper STT. Please speak into your microphone for 5 seconds...")
    try:
        res = stt.transcribe_from_mic(duration=5.0)
        print("\n--- Transcription Result ---")
        print("Text:", res["text"])
        print("Language:", res["language"])
        print("----------------------------")
    except Exception as e:
        print("Error during test:", e)

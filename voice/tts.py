"""
Silent Doctor — Text-to-Speech Module
=======================================
Local TTS using Coqui TTS (XTTS v2 for multilingual support).

Usage:
    tts = TextToSpeech()
    tts.speak("Hello, how can I help you?", language="en")
    tts.save_to_file("مرحبا", "output.wav", language="ar")
"""

from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import AUDIO_DIR, AUDIO_SAMPLE_RATE, TTS_MODEL_NAME
from utils.helpers import setup_logger

logger = setup_logger(__name__)


class TextToSpeech:
    """
    Local text-to-speech using Coqui TTS.

    Uses XTTS v2 for multilingual support including Arabic.
    Falls back to saving WAV files if audio playback is unavailable.
    """

    def __init__(
        self,
        model_name: str = TTS_MODEL_NAME,
    ):
        self.model_name = model_name
        self._tts = None

    @property
    def tts(self):
        """Lazy-load the TTS model on first use."""
        if self._tts is None:
            logger.info(f"Loading TTS model: {self.model_name}")
            try:
                from TTS.api import TTS as CoquiTTS

                self._tts = CoquiTTS(
                    model_name=self.model_name,
                    progress_bar=True,
                    gpu=False,  # CPU-only for low-resource
                )
                logger.info("✅ TTS model loaded.")
            except ImportError:
                logger.error(
                    "Coqui TTS not installed. "
                    "Install with: pip install TTS"
                )
                raise
        return self._tts

    def _map_language(self, language: str) -> str:
        """
        Map friendly language names to TTS language codes.
        XTTS v2 supports: en, es, fr, de, it, pt, pl, tr, ru, nl,
        cs, ar, zh-cn, ja, hu, ko
        """
        lang_map = {
            "english": "en",
            "arabic": "ar",
            "darija": "ar",    # Use Arabic as closest for Darija
            "amazigh": "fr",   # Fallback to French for Amazigh
            "french": "fr",
            "en": "en",
            "ar": "ar",
            "fr": "fr",
        }
        return lang_map.get(language.lower(), "en")

    def synthesize(
        self,
        text: str,
        language: str = "english",
        speaker_wav: Optional[str] = None,
    ) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: Text to convert to speech.
            language: Target language for speech.
            speaker_wav: Optional path to a reference speaker WAV
                         for voice cloning.

        Returns:
            Numpy array of audio samples.
        """
        tts_lang = self._map_language(language)
        logger.info(f"🔊 Synthesizing speech ({tts_lang}): '{text[:60]}...'")

        wav = self.tts.tts(
            text=text,
            language=tts_lang,
            speaker_wav=speaker_wav,
        )

        return np.array(wav, dtype=np.float32)

    def speak(
        self,
        text: str,
        language: str = "english",
        speaker_wav: Optional[str] = None,
    ):
        """
        Synthesize and immediately play audio through speakers.

        Falls back to saving a file if playback fails.
        """
        audio = self.synthesize(text, language, speaker_wav)

        try:
            import sounddevice as sd

            sd.play(audio, samplerate=22050)
            sd.wait()
            logger.info("🔊 Audio playback complete.")
        except Exception as exc:
            logger.warning(
                f"Cannot play audio ({exc}). Saving to file instead."
            )
            output_path = AUDIO_DIR / "last_output.wav"
            self.save_to_file(text, str(output_path), language, speaker_wav)

    def save_to_file(
        self,
        text: str,
        output_path: str,
        language: str = "english",
        speaker_wav: Optional[str] = None,
    ) -> str:
        """
        Synthesize speech and save to a WAV file.

        Returns:
            Absolute path to the saved WAV file.
        """
        tts_lang = self._map_language(language)
        logger.info(f"💾 Generating WAV: {output_path}")

        self.tts.tts_to_file(
            text=text,
            file_path=output_path,
            language=tts_lang,
            speaker_wav=speaker_wav,
        )

        logger.info(f"✅ Audio saved to {output_path}")
        return str(Path(output_path).resolve())


# ── Convenience function ────────────────────────────────────────────────

_cached_tts: Optional[TextToSpeech] = None


def get_tts(**kwargs) -> TextToSpeech:
    """Get or create a cached TextToSpeech instance."""
    global _cached_tts
    if _cached_tts is None:
        _cached_tts = TextToSpeech(**kwargs)
    return _cached_tts

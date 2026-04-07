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
import os

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
        if self._tts is None:
            logger.info(f"Loading gTTS wrapper")
            try:
                from gtts import gTTS
                self._tts = gTTS  # Just store the class reference
                logger.info("✅ gTTS loaded.")
            except ImportError:
                logger.error(
                    "gTTS not installed. "
                    "Install with: pip install gTTS"
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

        tts_lang = self._map_language(language)
        logger.info(f"🔊 Synthesizing speech array is not supported in gTTS natively. Saving/Playing automatically soon.")
        return np.array([])

    def speak(
        self,
        text: str,
        language: str = "english",
        speaker_wav: Optional[str] = None,
    ):
        """
        Synthesize and immediately play audio through the OS default player.
        """
        output_path = AUDIO_DIR / "last_output.mp3"
        self.save_to_file(text, str(output_path), language, speaker_wav)
        
        try:
            if os.name == 'nt':
                os.startfile(str(output_path))
            else:
                os.system(f"xdg-open '{output_path}'")
            logger.info("🔊 Audio playback complete.")
        except Exception as exc:
            logger.warning(
                f"Cannot play audio ({exc}). File saved cleanly."
            )

    def save_to_file(
        self,
        text: str,
        output_path: str,
        language: str = "english",
        speaker_wav: Optional[str] = None,
    ) -> str:
        """
        Synthesize speech and save to an MP3/WAV file using gTTS.

        Returns:
            Absolute path to the saved audio file.
        """
        tts_lang = self._map_language(language)
        logger.info(f"💾 Generating Audio: {output_path}")

        tts_obj = self.tts(text=text, lang=tts_lang)
        tts_obj.save(output_path)

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

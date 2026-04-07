"""
Silent Doctor — Translation Module
====================================
Local translation using NLLB (No Language Left Behind) via CTranslate2.

Supports: Darija, Amazigh, Arabic, English, French.

Usage:
    translator = Translator()
    result = translator.translate("Hello", src="english", tgt="darija")
"""

from pathlib import Path
from typing import Optional

from config.settings import LANGUAGE_CODES, NLLB_DEVICE, NLLB_MODEL
from utils.helpers import setup_logger

logger = setup_logger(__name__)


class Translator:
    """
    Offline translation using NLLB-200 via CTranslate2.

    Uses the distilled 600M parameter version for a good
    balance between quality and resource usage (~1.2 GB).
    """

    def __init__(
        self,
        model_name: str = NLLB_MODEL,
        device: str = NLLB_DEVICE,
    ):
        self.model_name = model_name
        self.device = device
        self._translator = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load the NLLB model and tokenizer."""
        if self._translator is not None:
            return

        logger.info(f"Loading NLLB translation model: {self.model_name}")

        try:
            import ctranslate2
            from transformers import AutoTokenizer

            # Convert HuggingFace model to CTranslate2 format if needed
            # For first run, the model will be downloaded and converted
            ct2_model_path = Path.home() / ".cache" / "silent_doctor" / "nllb_ct2"

            if not ct2_model_path.exists():
                logger.info(
                    "Converting NLLB model to CTranslate2 format "
                    "(one-time operation) ..."
                )
                ct2_model_path.mkdir(parents=True, exist_ok=True)

                import subprocess
                subprocess.run(
                    [
                        "ct2-nllb-converter",
                        "--model", self.model_name,
                        "--output_dir", str(ct2_model_path),
                        "--quantization", "int8",
                    ],
                    check=True,
                )
                logger.info("✅ Model conversion complete.")

            self._translator = ctranslate2.Translator(
                str(ct2_model_path),
                device=self.device,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("✅ NLLB model loaded.")

        except ImportError as e:
            logger.error(
                f"Required packages not installed: {e}. "
                "Install with: pip install ctranslate2 transformers sentencepiece"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load NLLB model: {e}")
            raise

    def _resolve_lang_code(self, language: str) -> str:
        """Resolve a friendly language name to an NLLB language code."""
        lang_lower = language.lower().strip()
        if lang_lower in LANGUAGE_CODES:
            return LANGUAGE_CODES[lang_lower]
        # Assume it's already an NLLB code
        return language

    def translate(
        self,
        text: str,
        src_lang: str = "english",
        tgt_lang: str = "darija",
    ) -> str:
        """
        Translate text between supported languages.

        Args:
            text: Input text to translate.
            src_lang: Source language (name or NLLB code).
            tgt_lang: Target language (name or NLLB code).

        Returns:
            Translated text string.
        """
        self._load_model()

        src_code = self._resolve_lang_code(src_lang)
        tgt_code = self._resolve_lang_code(tgt_lang)

        logger.info(f"🌐 Translating: {src_code} → {tgt_code}")

        # Tokenize with source language prefix
        self._tokenizer.src_lang = src_code
        tokens = self._tokenizer(text, return_tensors=None)
        input_ids = tokens["input_ids"]

        # Convert token IDs to token strings
        source_tokens = self._tokenizer.convert_ids_to_tokens(input_ids)

        # Translate
        target_prefix = [tgt_code]
        results = self._translator.translate_batch(
            [source_tokens],
            target_prefix=[target_prefix],
        )

        # Decode output tokens
        output_tokens = results[0].hypotheses[0]
        # Remove language prefix token
        if output_tokens and output_tokens[0] == tgt_code:
            output_tokens = output_tokens[1:]

        translated_text = self._tokenizer.decode(
            self._tokenizer.convert_tokens_to_ids(output_tokens),
            skip_special_tokens=True,
        )

        logger.info(f"✅ Translation complete: '{translated_text[:80]}...'")
        return translated_text

    def translate_to_english(self, text: str, src_lang: str) -> str:
        """Shortcut: translate from any supported language to English."""
        return self.translate(text, src_lang=src_lang, tgt_lang="english")

    def translate_from_english(self, text: str, tgt_lang: str) -> str:
        """Shortcut: translate from English to any supported language."""
        return self.translate(text, src_lang="english", tgt_lang=tgt_lang)


# ── Convenience function ────────────────────────────────────────────────

_cached_translator: Optional[Translator] = None


def get_translator(**kwargs) -> Translator:
    """Get or create a cached Translator instance."""
    global _cached_translator
    if _cached_translator is None:
        _cached_translator = Translator(**kwargs)
    return _cached_translator

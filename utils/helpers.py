"""
Silent Doctor — Shared Utilities
=================================
Image loading, audio recording, logging setup, and common helpers.
"""

import logging
import sys
from pathlib import Path

from typing import Optional

import numpy as np
from PIL import Image

from config.settings import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    IMAGE_SIZE,
    LOG_FORMAT,
    LOG_LEVEL,
)


# ── Logging ─────────────────────────────────────────────────────────────

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Create a configured logger for any module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level or LOG_LEVEL))
    return logger


logger = setup_logger("silent_doctor")


# ── Image Utilities ─────────────────────────────────────────────────────

def load_image(image_path: str) -> Image.Image:
    """
    Load an image from disk and convert to RGB.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        PIL Image in RGB mode.

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the file is not a valid image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as exc:
        raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc


def resize_image(
    img: Image.Image,
    size: tuple[int, int] = IMAGE_SIZE,
) -> Image.Image:
    """Resize an image to the target dimensions (default 224×224)."""
    return img.resize(size, Image.LANCZOS)


# ── Audio Utilities ─────────────────────────────────────────────────────

def record_audio(
    duration_seconds: float = 5.0,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    channels: int = AUDIO_CHANNELS,
) -> np.ndarray:
    """
    Record audio from the default microphone.

    Args:
        duration_seconds: How long to record.
        sample_rate: Sample rate in Hz (default 16 kHz for Whisper).
        channels: Number of audio channels (default mono).

    Returns:
        Numpy array of recorded audio samples (float32).
    """
    try:
        import sounddevice as sd
    except ImportError:
        logger.error(
            "sounddevice is not installed. "
            "Install it with: pip install sounddevice"
        )
        raise

    logger.info(f"🎤 Recording for {duration_seconds}s ...")
    audio = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
    )
    sd.wait()
    logger.info("🎤 Recording complete.")
    return audio.flatten()


def save_audio(
    audio: np.ndarray,
    output_path: str,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> str:
    """
    Save a numpy audio array to a WAV file.

    Returns:
        The absolute path to the saved file.
    """
    from scipy.io import wavfile

    # Convert float32 [-1, 1] to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, audio_int16)
    logger.info(f"💾 Audio saved to {output_path}")
    return str(Path(output_path).resolve())


# ── General Helpers ─────────────────────────────────────────────────────

def safe_import(module_name: str):
    """
    Attempt to import a module; return None with a warning if unavailable.
    Useful for optional heavy dependencies.
    """
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        logger.warning(f"Optional module '{module_name}' not installed.")
        return None


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

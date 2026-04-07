"""
Silent Doctor — Centralized Configuration
==========================================
All paths, model parameters, and runtime settings are defined here.
Models are loaded lazily — this file only stores paths and constants.
"""

import os
from pathlib import Path


# ── Project root (two levels up from config/settings.py) ────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Directory paths ─────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models" / "weights"
DATASETS_DIR = PROJECT_ROOT / "datasets"
FAISS_INDEX_DIR = PROJECT_ROOT / "rag" / "index"
AUDIO_DIR = PROJECT_ROOT / "audio"

# Ensure critical directories exist
for _dir in (MODELS_DIR, DATASETS_DIR, FAISS_INDEX_DIR, AUDIO_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ── Ollama (Local LLM) ─────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma:2b")


# ── Vision Models ───────────────────────────────────────────────────────
SKIN_MODEL_PATH = MODELS_DIR / "skin_mobilenetv2.pth"
EYE_MODEL_PATH = MODELS_DIR / "eye_mobilenetv3.pth"

IMAGE_SIZE = (224, 224)  # Standard input size for MobileNet

SKIN_CLASSES = ["acne", "eczema", "psoriasis", "melanoma", "normal_skin"]
EYE_CLASSES = ["conjunctivitis", "cataract", "dry_eye", "normal_eye"]


# ── Voice Pipeline ──────────────────────────────────────────────────────
# Whisper model size: tiny, base, small, medium, large-v3
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = "cpu"  # Use "cuda" if GPU available
WHISPER_COMPUTE_TYPE = "int8"  # Quantized for low-resource

# NLLB Translation model
NLLB_MODEL = os.getenv(
    "NLLB_MODEL",
    "facebook/nllb-200-distilled-600M",
)
NLLB_DEVICE = "cpu"

# Language codes for NLLB
LANGUAGE_CODES = {
    "darija": "ary_Arab",
    "amazigh": "ber_Latn",
    "arabic": "arb_Arab",
    "english": "eng_Latn",
    "french": "fra_Latn",
}

# Coqui TTS model
TTS_MODEL_NAME = os.getenv(
    "TTS_MODEL_NAME",
    "tts_models/multilingual/multi-dataset/xtts_v2",
)

# Audio settings
AUDIO_SAMPLE_RATE = 16000  # 16 kHz for Whisper
AUDIO_CHANNELS = 1


# ── RAG System ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "medical_index.faiss"
FAISS_METADATA_PATH = FAISS_INDEX_DIR / "medical_metadata.pkl"

# Chunk settings for document ingestion
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks
RAG_TOP_K = 5  # number of documents to retrieve


# ── Medical Prompt Templates ───────────────────────────────────────────
MEDICAL_SYSTEM_PROMPT = """You are Silent Doctor, a local medical assistant AI.
You help people in rural areas who have limited access to healthcare.

IMPORTANT RULES:
- Never provide a definitive diagnosis. Always recommend seeing a doctor.
- Be empathetic and use simple, clear language.
- If a condition sounds serious, urge the user to seek emergency care.
- Base your answers on the provided medical context when available.
- If you are unsure, say so honestly.
- Always remind the user that you are an AI assistant, not a real doctor.
"""

CHILD_HEALTH_SYSTEM_PROMPT = """You are Silent Doctor's child health module.
You help parents in rural areas assess common childhood health issues.

IMPORTANT RULES:
- Focus on common childhood conditions: fever, rash, eye infections, skin irritations.
- Always recommend professional medical care for serious symptoms.
- Provide comfort and first-aid guidance when appropriate.
- Use simple, reassuring language.
- Never replace professional pediatric care.
- Always remind parents you are an AI assistant.
"""


# ── Logging ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s"

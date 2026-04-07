"""
model.py – Load the trained Keras model and run predictions.
"""
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# ── Class labels ─────────────────────────────────────────────────────────────
LESION_CLASSES = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevi',
    5: 'Melanoma',
    6: 'Vascular lesions',
}

# Short code → long label mapping (same order as training)
LESION_SHORT = {
    0: 'akiec',
    1: 'bcc',
    2: 'bkl',
    3: 'df',
    4: 'nv',
    5: 'mel',
    6: 'vasc',
}

import os
from pathlib import Path

MODEL_PATH = str(Path(__file__).parent / "weights" / "mobilenet_model.h5")
IMG_SIZE   = (75, 100)              # (height, width) – matches your training setup

# ── Normalization constants (computed from the actual training split) ─────────
TRAIN_MEAN = 159.883804   # np.mean(x_train) before normalization
TRAIN_STD  = 46.454494    # np.std(x_train)  before normalization

# ── Singleton model loader ────────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        print(f"[model] Loading '{MODEL_PATH}' …")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("[model] Model loaded successfully.")
    return _model

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes -> normalised numpy array (1, H, W, 3).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]))
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - TRAIN_MEAN) / TRAIN_STD
    return arr.reshape(1, *IMG_SIZE, 3)

# ── Prediction ────────────────────────────────────────────────────────────────
def predict_image(image_bytes: bytes, top_k: int = 3) -> dict:
    """
    Run inference and return the top-k predictions plus a Grad-CAM heatmap.
    Returns a dict with 'top_k' list and 'heatmap_base64' string.
    """
    processed = preprocess_image(image_bytes)
    model     = get_model()
    probs     = model.predict(processed, verbose=0)[0]

    top_indices = np.argsort(probs)[::-1][:top_k]
    top_k_results = []
    for rank, idx in enumerate(top_indices, start=1):
        top_k_results.append({
            "rank":       rank,
            "label":      LESION_CLASSES[idx],
            "code":       LESION_SHORT[idx],
            "confidence": float(round(float(probs[idx]) * 100, 2)),
        })

    top_idx = int(top_indices[0])

    # Grad-CAM heatmap
    heatmap_b64 = ""
    try:
        from models.visualization import gradcam_keras, overlay_heatmap, _pil_to_bytes
        from PIL import Image
        import io
        # Find the last Conv layer name in the model
        last_conv_name = None
        import tensorflow as tf
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_name = layer.name
                break
        if last_conv_name:
            cam = gradcam_keras(model, processed, last_conv_name, top_idx)
            orig_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((256, 256))
            overlay = overlay_heatmap(orig_pil, cam)
            heatmap_b64 = _pil_to_bytes(overlay)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Skin Grad-CAM failed: {e}")

    return {
        "top_k": top_k_results,
        "heatmap_base64": heatmap_b64,
    }


class SkinModelWrapper:
    def predict(self, image_path: str) -> dict:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return predict_image(image_bytes)

_cached_skin_model = None

def get_skin_model():
    """Get or create cached SkinModel adapter instance."""
    global _cached_skin_model
    if _cached_skin_model is None:
        _cached_skin_model = SkinModelWrapper()
    return _cached_skin_model

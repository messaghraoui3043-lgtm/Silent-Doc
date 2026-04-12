"""
visualization.py - Grad-CAM heatmap generation for both PyTorch and TensorFlow/Keras models.

Provides:
  - gradcam_pytorch(): For ResNet18 Eye Disease model (PyTorch)
  - gradcam_keras(): For MobileNetV2 Skin Lesion model (TensorFlow/Keras)
  - overlay_heatmap(): Blends the colorized heatmap over the original image
"""

import cv2
import numpy as np
import base64
import io
from PIL import Image


# ── Shared Utilities ──────────────────────────────────────────────────────────

def _pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> str:
    """Convert a PIL image to a base64-encoded string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def overlay_heatmap(original_pil: Image.Image, raw_cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Overlay a raw Grad-CAM saliency map over the original PIL image.

    Args:
        original_pil: The source image (any size, RGB).
        raw_cam: A 2D NumPy float32 array of gradient activations (values 0–1).
        alpha: Heatmap opacity (0-1). Defaults to 0.45.

    Returns:
        A PIL Image with the colored heatmap blended over the original.
    """
    w, h = original_pil.size
    
    # Normalize and colorize the CAM
    cam = np.clip(raw_cam, 0, 1)
    cam_uint8 = np.uint8(255 * cam)
    cam_resized = cv2.resize(cam_uint8, (w, h))
    heatmap_bgr = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    # Blend over the original image
    orig_np = np.array(original_pil.convert("RGB"), dtype=np.float32)
    heat_np = heatmap_rgb.astype(np.float32)
    blended = (1 - alpha) * orig_np + alpha * heat_np
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)


# ── PyTorch Grad-CAM (Eye Model - ResNet18) ──────────────────────────────────

def gradcam_pytorch(model, tensor_input, target_layer, target_class_idx: int) -> np.ndarray:
    """
    Compute Grad-CAM for a PyTorch model using forward + backward hooks.

    Args:
        model: The PyTorch model (eval mode).
        tensor_input: Preprocessed tensor of shape (1, C, H, W).
        target_layer: The convolutional layer to attach hooks to (e.g. model.network.layer4[-1]).
        target_class_idx: The integer class index to explain.

    Returns:
        A 2D NumPy float32 array (normalized CAM).
    """
    activations = {}
    gradients = {}

    def save_activation(module, input, output):
        activations["value"] = output.detach()

    def save_gradient(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    fwd_hook = target_layer.register_forward_hook(save_activation)
    bwd_hook = target_layer.register_full_backward_hook(save_gradient)

    try:
        import torch
        model.eval()
        output = model(tensor_input)
        model.zero_grad()
        class_score = output[0, target_class_idx]
        class_score.backward()

        grads = gradients["value"][0]          # (C, H, W)
        acts = activations["value"][0]          # (C, H, W)

        weights = grads.mean(dim=(1, 2))        # Global Average Pool → (C,)
        cam = torch.zeros(acts.shape[1:])       # (H, W)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = torch.relu(cam).numpy()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.astype(np.float32)

    finally:
        fwd_hook.remove()
        bwd_hook.remove()


# ── TensorFlow/Keras Grad-CAM (Skin Model - MobileNetV2) ────────────────────

def gradcam_keras(keras_model, img_array: np.ndarray, target_layer_name: str, target_class_idx: int) -> np.ndarray:
    """
    Compute Grad-CAM for a TensorFlow/Keras model using GradientTape.

    Args:
        keras_model: The compiled Keras model.
        img_array: Preprocessed image as numpy array of shape (1, H, W, 3).
        target_layer_name: Name of the final convolutional layer.
        target_class_idx: The integer class index to explain.

    Returns:
        A 2D NumPy float32 array (normalized CAM).
    """
    import tensorflow as tf

    # Build a sub-model that outputs (conv_output, predictions)
    grad_model = tf.keras.models.Model(
        inputs=keras_model.inputs,
        outputs=[keras_model.get_layer(target_layer_name).output, keras_model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        class_score = predictions[:, target_class_idx]

    grads = tape.gradient(class_score, conv_outputs)   # (1, H, W, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_outputs = conv_outputs[0]                          # (H, W, C)
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)  # (H, W)
    cam = tf.nn.relu(cam).numpy()

    if cam.max() > 0:
        cam /= cam.max()
    return cam.astype(np.float32)
    

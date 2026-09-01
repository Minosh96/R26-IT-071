"""
inference/view_validator.py
===========================
Vehicle View Classifier – MobileNetV3Small (5 classes)
Trained weights: vehicle_view_weights.weights.h5

Classes (in training order):
    0 – Front
    1 – Rear
    2 – Left
    3 – Right
    4 – Roof   (top/up)

Usage:
    from inference.view_validator import load_view_model, validate_view

    view_model = load_view_model()                         # once at startup
    result     = validate_view(view_model, image_bytes, expected_view="front")
    # result = {"valid": True/False, "predicted": "Front", "confidence": 97.3,
    #           "expected": "Front", "message": "..."}
"""

import os
import io
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from PIL import Image

# ── Class labels (must match training label indices) ─────────────────────────
VIEW_LABELS = ["Front", "Rear", "Left", "Right", "Roof"]

# Map API view name → expected label index
VIEW_NAME_TO_IDX = {
    "front": 0,
    "rear":  1,
    "left":  2,
    "right": 3,
    "roof":  4,
    "up":    4,   # alternate name used by some endpoints
}

# Confidence threshold: if model confidence < this, treat as "uncertain" but
# do NOT block the user — just warn.
CONFIDENCE_THRESHOLD = 0.40

# Component directory (two levels up from this file)
_COMPONENT_DIR = pathlib.Path(__file__).parent.parent.absolute()


# ── Model architecture ────────────────────────────────────────────────────────

def _build_view_model() -> Model:
    """
    Reconstructs the MobileNetV3Small view classifier.

    Architecture matches the training script:
      Input (224×224×3) → [optional aug, skipped at inference]
      → MobileNetV3Small (include_preprocessing=True)
      → GlobalAveragePooling2D
      → Dense(256, relu)
      → Dense(5, softmax)

    Weights are loaded separately via load_view_model().
    """
    inputs = tf.keras.Input(shape=(224, 224, 3), name="view_input")

    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None,                 # weights loaded manually
        include_preprocessing=True,   # built-in [0,255] → [-1,1] normalisation
        pooling=None,
    )
    base.trainable = False            # frozen at inference

    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense")(x)
    out = layers.Dense(len(VIEW_LABELS), activation="softmax", name="view_output")(x)

    return Model(inputs=inputs, outputs=out, name="vehicle_view_classifier")


# ── Weight loading ────────────────────────────────────────────────────────────

def _find_weights_file(filename: str = "vehicle_view_weights.weights.h5") -> str:
    """Search for the weights file in standard locations."""
    search_paths = [
        _COMPONENT_DIR / filename,
        pathlib.Path(filename),
        pathlib.Path("component2-body-condition") / filename,
    ]
    for p in search_paths:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Vehicle view weights file '{filename}' not found. Searched:\n"
        + "\n".join(f"  {p}" for p in search_paths)
    )


def load_view_model() -> Model:
    """
    Builds the view classifier and loads weights from the .h5 file.

    Returns: compiled Keras Model ready for inference.
    """
    weights_path = _find_weights_file()
    print(f"[VIEW] Loading vehicle view model weights from:\n       {weights_path}")

    model = _build_view_model()

    # Warm-up pass so all variable shapes are materialised before load_weights
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model(dummy, training=False)

    # Load weights – uses keras load_weights (supports .weights.h5 format)
    model.load_weights(weights_path)

    # Sanity-check prediction
    test_pred = model(dummy, training=False).numpy()[0]
    print(
        f"[VIEW] View model loaded OK. "
        f"Dummy prediction: {dict(zip(VIEW_LABELS, [round(float(p)*100, 1) for p in test_pred]))}"
    )
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes → model-ready (1, 224, 224, 3) float32 array."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224), Image.BILINEAR)
    arr   = np.array(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


# ── Public validation API ─────────────────────────────────────────────────────

def validate_view(
    model: Model,
    image_bytes: bytes,
    expected_view: str,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Validates that *image_bytes* actually shows the *expected_view* of a vehicle.

    Parameters
    ----------
    model          : Loaded view classifier (from load_view_model()).
    image_bytes    : Raw bytes of the uploaded image.
    expected_view  : One of 'front', 'rear', 'left', 'right', 'roof', 'up'.
    confidence_threshold : Below this the result is marked as uncertain.

    Returns
    -------
    dict with keys:
        valid       – bool   True if predicted view matches expected view.
        predicted   – str    e.g. "Left"
        expected    – str    e.g. "Left"  (human-readable)
        confidence  – float  0–100 percentage confidence of top prediction
        all_probs   – dict   {label: confidence%} for all 5 views
        uncertain   – bool   True if confidence < threshold (show a softer warning)
        message     – str    Human-readable verdict message for the UI
    """
    normalized = expected_view.lower().strip()
    expected_idx = VIEW_NAME_TO_IDX.get(normalized)
    if expected_idx is None:
        raise ValueError(f"Unknown view name: '{expected_view}'. "
                         f"Valid options: {list(VIEW_NAME_TO_IDX.keys())}")

    expected_label = VIEW_LABELS[expected_idx]

    # Run inference
    arr   = _preprocess(image_bytes)
    probs = model(arr, training=False).numpy()[0]   # shape (5,)

    predicted_idx   = int(np.argmax(probs))
    predicted_label = VIEW_LABELS[predicted_idx]
    confidence      = float(probs[predicted_idx]) * 100.0

    is_correct  = (predicted_idx == expected_idx)
    is_uncertain = confidence < (confidence_threshold * 100)

    all_probs = {label: round(float(probs[i]) * 100, 2)
                 for i, label in enumerate(VIEW_LABELS)}

    if is_correct:
        message = (
            f"✅ Correct view detected: {predicted_label} "
            f"({confidence:.1f}% confidence)."
        )
    elif is_uncertain:
        message = (
            f"⚠️ View is unclear (confidence: {confidence:.1f}%). "
            f"This looks most like a {predicted_label} view, but you requested "
            f"{expected_label}. Please ensure the image clearly shows the "
            f"{expected_label} of the vehicle."
        )
    else:
        message = (
            f"❌ Incorrect view detected! This image appears to be a "
            f"{predicted_label} view ({confidence:.1f}% confidence), "
            f"but you uploaded it as the {expected_label} view. "
            f"Please upload the correct {expected_label} image."
        )

    return {
        "valid":       is_correct or is_uncertain,   # uncertain = soft-pass
        "correct":     is_correct,
        "predicted":   predicted_label,
        "expected":    expected_label,
        "confidence":  round(confidence, 2),
        "all_probs":   all_probs,
        "uncertain":   is_uncertain,
        "message":     message,
    }


def validate_all_views(
    model: Model,
    views: dict,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Validates all 5 views in one call.

    Parameters
    ----------
    model  : Loaded view classifier.
    views  : {view_name: image_bytes}  e.g. {"front": b"..."}

    Returns
    -------
    dict:
        all_valid       – bool    True only if every view passed validation.
        validation_errors – list  List of error messages for failed views.
        view_results    – dict    Per-view validation result dicts.
    """
    view_results = {}
    errors       = []
    warnings     = []

    for view_name, image_bytes in views.items():
        result = validate_view(model, image_bytes, view_name, confidence_threshold)
        view_results[view_name] = result

        if not result["correct"]:
            if result["uncertain"]:
                warnings.append(result["message"])
            else:
                errors.append({
                    "view":     view_name,
                    "message":  result["message"],
                    "expected": result["expected"],
                    "detected": result["predicted"],
                })

    return {
        "all_valid":          len(errors) == 0,
        "has_warnings":       len(warnings) > 0,
        "validation_errors":  errors,
        "validation_warnings": warnings,
        "view_results":       view_results,
    }

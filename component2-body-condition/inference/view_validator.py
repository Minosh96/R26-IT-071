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
from PIL import Image, ImageOps

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

# 13 Detailed body parts mapped to their primary view
BODY_PARTS = [
    'Bonnet', 'Dicky_Door', 'Front_Bumper', 'Left_Front_Door', 'Left_Front_Fender',
    'Left_Rear_Door', 'Left_Rear_Fender', 'Rear_Bumper', 'Right_Front_Door',
    'Right_Front_Fender', 'Right_Rear_Door', 'Right_Rear_Fender', 'Roof'
]

PART_TO_VIEW_MAP = {
    'Bonnet': 'Front', 'Front_Bumper': 'Front',
    'Dicky_Door': 'Rear', 'Rear_Bumper': 'Rear',
    'Left_Front_Door': 'Left', 'Left_Front_Fender': 'Left', 'Left_Rear_Door': 'Left', 'Left_Rear_Fender': 'Left',
    'Right_Front_Door': 'Right', 'Right_Front_Fender': 'Right', 'Right_Rear_Door': 'Right', 'Right_Rear_Fender': 'Right',
    'Roof': 'Roof'
}

# Confidence threshold: below this the result is marked as uncertain
CONFIDENCE_THRESHOLD = 0.40

# Component directory (two levels up from this file)
_COMPONENT_DIR = pathlib.Path(__file__).parent.parent.absolute()

# ── Vehicle Detector (ImageNet backbone for rejecting non-car images) ────────
_VEHICLE_DETECTOR = None

VEHICLE_KEYWORDS = (
    "car", "wagon", "cab", "minivan", "jeep", "truck", "van", "wheel", "grille",
    "convertible", "limousine", "racer", "vehicle", "bumper", "pickup", "motor",
    "bus", "ambulance", "fire_engine", "tow_truck", "garbage_truck", "police_van",
    "recreational_vehicle", "trailer_truck", "snowplow", "go-kart", "golfcart",
    "sports_car", "sedan", "coupe", "suv"
)

def _get_vehicle_detector():
    """Lazily loads pre-trained ImageNet MobileNetV3Small to verify vehicle presence."""
    global _VEHICLE_DETECTOR
    if _VEHICLE_DETECTOR is None:
        try:
            _VEHICLE_DETECTOR = tf.keras.applications.MobileNetV3Small(
                weights="imagenet", include_top=True
            )
            # Warm up
            _VEHICLE_DETECTOR(np.zeros((1, 224, 224, 3), dtype=np.float32), training=False)
        except Exception as e:
            print(f"[VIEW] Warning: Vehicle detector model could not be loaded: {e}")
            _VEHICLE_DETECTOR = False
    return _VEHICLE_DETECTOR if _VEHICLE_DETECTOR is not False else None


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

def _preprocess(image_bytes: bytes) -> tuple:
    """
    Convert raw image bytes → model-ready (1, 224, 224, 3) float32 array.
    Automatically applies EXIF transpose so camera orientation from phones
    (portrait / rotated) is preserved correctly.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    resized = image.resize((224, 224), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32)
    return np.expand_dims(arr, axis=0), image


# ── Public validation API ─────────────────────────────────────────────────────

def validate_view(
    model: Model,
    image_bytes: bytes,
    expected_view: str,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    part_model: Model = None,
) -> dict:
    """
    Validates that *image_bytes* actually shows the *expected_view* of a vehicle.
    Also verifies whether the image is actually a vehicle image (rejecting non-car images),
    and ensembles with part_model if available to prevent Left/Right confusion.

    Parameters
    ----------
    model          : Loaded view classifier (from load_view_model()).
    image_bytes    : Raw bytes of the uploaded image.
    expected_view  : One of 'front', 'rear', 'left', 'right', 'roof', 'up'.
    confidence_threshold : Below this the result is marked as uncertain.
    part_model     : Optional loaded EfficientNetV2B0 part classifier.

    Returns
    -------
    dict with keys:
        valid       – bool   True if predicted view matches expected view.
        correct     – bool
        is_vehicle  – bool   False if image does not show a vehicle.
        predicted   – str    e.g. "Left" or "Non-Vehicle"
        expected    – str    e.g. "Left"  (human-readable)
        confidence  – float  0–100 percentage confidence of top prediction
        all_probs   – dict   {label: confidence%} for all 5 views
        uncertain   – bool   True if confidence < threshold
        message     – str    Human-readable verdict message for the UI
    """
    normalized = expected_view.lower().strip()
    expected_idx = VIEW_NAME_TO_IDX.get(normalized)
    if expected_idx is None:
        raise ValueError(f"Unknown view name: '{expected_view}'. "
                         f"Valid options: {list(VIEW_NAME_TO_IDX.keys())}")

    expected_label = VIEW_LABELS[expected_idx]

    # Preprocess image with EXIF auto-rotation
    arr, image = _preprocess(image_bytes)

    # ── 1. Non-car / Out-of-Distribution Vehicle Verification ─────────────────
    detector = _get_vehicle_detector()
    is_vehicle = True

    part_probs = None
    part_max_conf = 0.0
    if part_model is not None:
        try:
            part_preds = part_model(arr, training=False).numpy()[0]
            part_probs = part_preds
            part_max_conf = float(np.max(part_preds))
        except Exception as pe:
            print(f"[VIEW] Warning: part_model evaluation error: {pe}")

    if detector is not None:
        try:
            from tensorflow.keras.applications.mobilenet_v3 import decode_predictions, preprocess_input
            in_preds = detector(preprocess_input(arr.copy()), training=False).numpy()
            top5 = decode_predictions(in_preds, top=5)[0]
            in_vehicle_score = sum(
                prob for syn, name, prob in top5
                if any(k in name.lower() for k in VEHICLE_KEYWORDS)
            )
            # A photo is recognized as a vehicle if ImageNet detects vehicle features
            # OR if our specialized 13-part body classifier recognizes a vehicle panel
            is_vehicle = (in_vehicle_score >= 0.08) or (part_max_conf >= 0.35)
        except Exception as de:
            print(f"[VIEW] Warning: vehicle detection check error: {de}")
            is_vehicle = True

    if not is_vehicle:
        return {
            "valid":       False,
            "correct":     False,
            "is_vehicle":  False,
            "predicted":   "Non-Vehicle",
            "expected":    expected_label,
            "confidence":  0.0,
            "all_probs":   {label: 0.0 for label in VIEW_LABELS},
            "uncertain":   False,
            "message":     (
                f"This does not appear to be a car-related image. "
                f"Please upload a clear photo of the {expected_label} view of the vehicle."
            ),
        }

    # ── 2. Vehicle View Inference & Panel Consensus ───────────────────────────
    view_probs = model(arr, training=False).numpy()[0]   # shape (5,)

    # If part_model is available, aggregate panel probabilities by view to eliminate Left/Right confusion
    if part_probs is not None:
        p_view = np.zeros(len(VIEW_LABELS), dtype=np.float32)
        for p_name, prob in zip(BODY_PARTS, part_probs):
            v_name = PART_TO_VIEW_MAP.get(p_name)
            if v_name:
                p_view[VIEW_NAME_TO_IDX[v_name.lower()]] += prob

        # 50/50 balanced ensemble between overall angle and panel-specific classifier
        final_probs = 0.5 * view_probs + 0.5 * p_view
    else:
        final_probs = view_probs

    predicted_idx   = int(np.argmax(final_probs))
    predicted_label = VIEW_LABELS[predicted_idx]
    confidence      = float(final_probs[predicted_idx]) * 100.0

    is_correct   = (predicted_idx == expected_idx)
    is_uncertain = confidence < (confidence_threshold * 100)

    all_probs = {label: round(float(final_probs[i]) * 100, 2)
                 for i, label in enumerate(VIEW_LABELS)}

    if is_correct and not is_uncertain:
        message = (
            f"Correct view detected: {predicted_label} "
            f"({confidence:.1f}% confidence)."
        )
        is_valid = True
    elif is_correct and is_uncertain:
        message = (
            f"Low confidence ({confidence:.1f}%). "
            f"This looks like {predicted_label}, but the angle is unclear. "
            f"Please ensure the vehicle is clearly framed."
        )
        is_valid = True
    elif is_uncertain:
        message = (
            f"View is unclear (confidence: {confidence:.1f}%). "
            f"This does not clearly match the {expected_label} view. "
            f"Please ensure the image clearly shows the {expected_label} of the vehicle."
        )
        is_valid = False
    else:
        message = (
            f"Incorrect view detected! This image appears to be a "
            f"{predicted_label} view ({confidence:.1f}% confidence), "
            f"but you uploaded it as the {expected_label} view. "
            f"Please upload the correct {expected_label} image."
        )
        is_valid = False

    return {
        "valid":       is_valid,
        "correct":     is_correct,
        "is_vehicle":  True,
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
    part_model: Model = None,
) -> dict:
    """
    Validates all 5 views in one call.

    Parameters
    ----------
    model        : Loaded view classifier.
    views        : {view_name: image_bytes}  e.g. {"front": b"..."}
    confidence_threshold : Below this the result is marked as uncertain.
    part_model   : Optional loaded part classifier.

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
        result = validate_view(
            model, image_bytes, view_name,
            confidence_threshold=confidence_threshold,
            part_model=part_model
        )
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


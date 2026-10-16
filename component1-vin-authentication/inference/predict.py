import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

from inference.vin_gate import looks_like_vin

MODEL_PATH = "models/vin_tampering_resnet50_finetuned.keras"

_model = None


def _get_model():
    """Load the tampering model on first use, so the service can start and
    reject non-VIN photos even before the model file is in place."""
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def predict_vin(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "label": "Invalid Image",
            "confidence": 0.0,
            "invalid_image": True,
        }

    # The tampering model can only answer "original" or "altered", so first
    # check that the photo actually looks like a VIN / chassis number plate.
    is_vin, vin_score = looks_like_vin(img)
    if not is_vin:
        return {
            "label": "Not a VIN",
            "confidence": 0.0,
            "not_vin": True,
            "vin_score": round(vin_score, 3),
        }

    # Resize image for the model
    img = cv2.resize(img, (224, 224))

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to model input format
    img = np.expand_dims(img, axis=0)

    # Apply ResNet50 preprocessing
    img = preprocess_input(img)

    # Probability that the VIN is original
    original_probability = float(_get_model().predict(img, verbose=0)[0][0])

    if original_probability > 0.70:
        label = "Original"
        confidence = original_probability

    elif original_probability < 0.30:
        label = "Altered"
        confidence = 1 - original_probability

    else:
        label = "Need Review"
        confidence = 1 - abs(original_probability - 0.5)

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "vin_score": round(vin_score, 3),
    }

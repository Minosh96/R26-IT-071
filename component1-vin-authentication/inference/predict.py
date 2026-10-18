import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as _resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as _mobilenet_preprocess

from inference.vin_gate import looks_like_vin

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RESNET_MODEL_PATH = os.path.join(_MODELS_DIR, "vin_tampering_resnet50_finetuned.keras")
MOBILENET_MODEL_PATH = os.path.join(_MODELS_DIR, "vin_tampering_mobilenetv2.keras")

# The fine-tuned ResNet50 model is git-ignored (*.keras) and may not exist in a
# fresh clone. Fall back to the committed MobileNetV2 model so the service still
# starts, using the preprocessing that matches whichever model was loaded.
if os.path.exists(RESNET_MODEL_PATH):
    MODEL_PATH = RESNET_MODEL_PATH
    preprocess_input = _resnet_preprocess
elif os.path.exists(MOBILENET_MODEL_PATH):
    print(f"[WARN] {RESNET_MODEL_PATH} not found - falling back to MobileNetV2 model.")
    MODEL_PATH = MOBILENET_MODEL_PATH
    preprocess_input = _mobilenet_preprocess
else:
    MODEL_PATH = RESNET_MODEL_PATH
    preprocess_input = _resnet_preprocess

_model = None


def _get_model():
    """Load the tampering model on first use, so the service can start and
    reject non-VIN photos even before the model file is in place."""
    global _model
    if _model is None:
        print(f"[INFO] Loading VIN model: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def predict_vin(image_bytes, detected_number=None):
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
    is_vin, vin_score = looks_like_vin(img, detected_number)
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

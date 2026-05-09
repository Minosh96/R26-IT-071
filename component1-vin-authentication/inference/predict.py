import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


MODEL_PATH = "models/vin_tampering_mobilenetv2.keras"

model = tf.keras.models.load_model(MODEL_PATH)


def predict_vin(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "label": "Invalid Image",
            "confidence": 0.0
        }

    # Resize image for MobileNetV2
    img = cv2.resize(img, (224, 224))

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to model input format
    img = np.expand_dims(img, axis=0)

    # Apply MobileNetV2 preprocessing
    img = preprocess_input(img)

    # Get prediction
    prediction = model.predict(img)[0][0]
    confidence_score = float(prediction)

    # Decide final label
    original_probability = float(model.predict(img)[0][0])

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
        "confidence": round(confidence, 2)
    }
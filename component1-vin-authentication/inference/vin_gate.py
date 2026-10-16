"""
Checks that an uploaded photo looks like a VIN / chassis number plate before the
tampering model is asked about it. The tampering model only knows two answers
(original or altered), so any other photo, such as a phone case or a car door,
would otherwise be forced into one of them.

How it works: each photo is turned into a feature vector by MobileNetV2
pretrained on ImageNet, and compared with the vectors of the VIN training images
(models/vin_reference_embeddings.npy). The score is the mean cosine similarity
to the 5 closest training images. Calibrated with tools/build_vin_reference.py:

  VIN photos of vehicles not in the reference set (data/val, 42)  0.69 and above
  car body photos (65)                                            up to 0.49
  wallpapers and landscapes (36)                                  up to 0.57
  phone screenshots and objects (8)                               up to 0.54
  pages of ordinary text (12)                                     up to 0.60

The threshold sits between the two groups and can be changed with VIN_GATE_MIN.
"""
import os

import cv2
import numpy as np

REFERENCE_PATH = "models/vin_reference_embeddings.npy"
VIN_GATE_MIN = float(os.getenv("VIN_GATE_MIN", 0.63))
TOP_K = 5

_extractor = None
_reference = None


def _load():
    global _extractor, _reference
    if _extractor is None:
        import tensorflow as tf
        _extractor = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3))
        _reference = np.load(REFERENCE_PATH).astype(np.float32)
    return _extractor, _reference


def embed(img_bgr):
    """Unit-length MobileNetV2 feature vector for a BGR image."""
    import tensorflow as tf
    extractor, _ = _load()
    x = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x[None])
    v = extractor.predict(x, verbose=0)[0]
    return v / (np.linalg.norm(v) + 1e-9)


def vin_likeness(img_bgr):
    """Mean cosine similarity to the closest VIN training images (0 to 1)."""
    _, reference = _load()
    similarities = reference @ embed(img_bgr)
    return float(np.sort(similarities)[-TOP_K:].mean())


def looks_like_vin(img_bgr):
    score = vin_likeness(img_bgr)
    return score >= VIN_GATE_MIN, score

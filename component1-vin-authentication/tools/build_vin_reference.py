"""
Builds models/vin_reference_embeddings.npy from the VIN training images and
reports how well the VIN-photo check separates VIN photos from other photos.

Run from component1-vin-authentication:
    python tools/build_vin_reference.py [folder of non-VIN photos ...]

The validation images (data/val, vehicles not in the training set) are scored
against the training reference, so the report shows how unseen VIN photos score.
Pass one or more folders of non-VIN photos (cars, people, rooms, screenshots,
documents) to see how they score.
"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import vin_gate  # noqa: E402

TRAIN = sorted(glob.glob("data/train/*/*.jpg"))
VAL = sorted(glob.glob("data/val/*/*.jpg"))


def load(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def main():
    import tensorflow as tf
    vin_gate._extractor = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", pooling="avg", input_shape=(224, 224, 3))
    reference = np.stack([vin_gate.embed(load(p)) for p in TRAIN]).astype(np.float16)
    np.save(vin_gate.REFERENCE_PATH, reference)
    vin_gate._reference = reference.astype(np.float32)
    print(f"Saved {len(TRAIN)} reference vectors to {vin_gate.REFERENCE_PATH}")

    val = np.array([vin_gate.vin_likeness(load(p)) for p in VAL])
    print(f"Unseen VIN photos (data/val): n={len(val)} min={val.min():.3f} median={np.median(val):.3f}")
    for folder in sys.argv[1:]:
        files = [f for f in glob.glob(os.path.join(folder, "*"))
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        scores = np.array([vin_gate.vin_likeness(load(f)) for f in files if load(f) is not None])
        print(f"{folder}: n={len(scores)} max={scores.max():.3f} median={np.median(scores):.3f}")
    print(f"Current threshold VIN_GATE_MIN = {vin_gate.VIN_GATE_MIN}")


if __name__ == "__main__":
    main()

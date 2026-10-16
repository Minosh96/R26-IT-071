"""Tests for the VIN service, focused on rejecting photos that are not a VIN plate."""
import glob
import io
import os

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app import app
from inference.predict import MODEL_PATH
from inference.vin_gate import looks_like_vin

client = TestClient(app)
VAL_IMAGES = sorted(glob.glob("data/val/*/*.jpg"))


def jpeg(img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr)
    assert ok
    return buf.tobytes()


def text_page():
    """A page of ordinary printed text: has text, but is not a VIN plate."""
    img = np.full((600, 900, 3), 245, np.uint8)
    for line in range(12):
        cv2.putText(img, "invoice total amount date name address phone", (30, 45 + line * 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    return img


def everyday_scene():
    """Coloured shapes on a gradient, standing in for a random everyday photo."""
    rng = np.random.default_rng(0)
    img = np.zeros((480, 640, 3), np.uint8)
    img[:] = np.linspace(40, 200, 640, dtype=np.uint8)[None, :, None]
    for _ in range(15):
        centre = tuple(int(v) for v in rng.integers(0, 480, 2))
        cv2.circle(img, centre, int(rng.integers(20, 90)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return img


def post(data, name="photo.jpg"):
    return client.post("/predict", files={"file": (name, io.BytesIO(data), "image/jpeg")})


def test_health():
    assert client.get("/health").status_code == 200


def test_unsupported_file_type():
    assert post(b"hello", "notes.txt").status_code == 400


def test_corrupt_image():
    assert post(os.urandom(5000)).status_code == 400


@pytest.mark.parametrize("make_image", [text_page, everyday_scene])
def test_non_vin_photo_is_rejected(make_image):
    r = post(jpeg(make_image()))
    assert r.status_code == 422
    assert r.json()["code"] == "NOT_VIN"


def test_unseen_vin_photos_pass_the_check():
    """VIN photos of vehicles that are not in the reference set must all pass."""
    assert VAL_IMAGES, "data/val images are missing"
    results = [looks_like_vin(cv2.imread(p))[0] for p in VAL_IMAGES]
    assert all(results), f"{results.count(False)} of {len(results)} VIN photos were rejected"


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="tampering model file not available")
def test_vin_photo_gets_a_verdict():
    with open(VAL_IMAGES[0], "rb") as f:
        r = post(f.read())
    assert r.status_code == 200
    assert r.json()["label"] in {"Original", "Altered", "Need Review"}

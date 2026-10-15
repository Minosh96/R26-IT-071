import os
import sys
import pathlib

import numpy as np
import pytest
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_AUDIO = ROOT / "data" / "test"

# The service uses paths relative to the component folder.
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("API_SECRET_TOKEN", "test-token")


@pytest.fixture(scope="session")
def app_module():
    from api import app as app_module  # loads YAMNet and the SVM once per test run
    return app_module


@pytest.fixture(scope="session")
def client(app_module):
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


@pytest.fixture(scope="session")
def auth(app_module):
    return {"Authorization": f"Bearer {app_module.API_TOKEN}"}


@pytest.fixture(scope="session")
def synthetic_audio(tmp_path_factory):
    """Non-engine clips: a pure tone and moderate white noise."""
    folder = tmp_path_factory.mktemp("synthetic")
    sr, n = 16000, 16000 * 5
    rng = np.random.default_rng(0)
    clips = {
        "tone": 0.4 * np.sin(2 * np.pi * 440 * np.arange(n) / sr),
        "white_noise": 0.3 * rng.standard_normal(n),
    }
    paths = {}
    for name, audio in clips.items():
        paths[name] = folder / f"{name}.wav"
        sf.write(paths[name], audio.astype(np.float32), sr)
    return paths


def audio(relative_path):
    """Open a file from data/test for a multipart upload."""
    path = TEST_AUDIO / relative_path
    return (open(path, "rb"), path.name)

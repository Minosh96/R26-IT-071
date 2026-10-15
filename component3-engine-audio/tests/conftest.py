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


@pytest.fixture(scope="session")
def full_recordings(tmp_path_factory):
    """
    30-second "single recordings" made by joining real clips: an engine start,
    then idle, then revving (or knocking). Saved as 16-bit WAV like the phone app.
    Returns {name: (path, start_end, idle_end)}.
    """
    import librosa
    folder = tmp_path_factory.mktemp("full")
    sr = 16000
    load = lambda name, seconds=None: librosa.load(TEST_AUDIO / name, sr=sr, duration=seconds)[0]
    start = load("healthy/CAG 5756_Start.wav")
    idle = load("healthy/CAQ 1654_Idle.wav", 12)
    revs = load("healthy/CAL 2467_High RPM.wav", 10)
    knock = load("knocking/Engine knock 6.wav", 10)
    start_end, idle_end = len(start) / sr, (len(start) + len(idle)) / sr
    noise = 0.3 * np.random.default_rng(1).standard_normal(sr * 25)
    clips = {
        "healthy": (np.concatenate([start, idle, revs]), start_end, idle_end),
        "knocking": (np.concatenate([start, idle, knock]), start_end, idle_end),
        "no_start": (np.concatenate([idle, revs]), None, None),
        "too_short": (np.concatenate([start, idle[:sr * 4]]), None, None),
        "white_noise": (noise, None, None),
    }
    out = {}
    for name, (signal, s_end, i_end) in clips.items():
        path = folder / f"{name}.wav"
        sf.write(path, signal.astype(np.float32), sr, subtype="PCM_16")
        out[name] = (path, s_end, i_end)
    return out


def audio(relative_path):
    """Open a file from data/test for a multipart upload."""
    path = TEST_AUDIO / relative_path
    return (open(path, "rb"), path.name)

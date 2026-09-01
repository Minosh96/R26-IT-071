"""
This module provides the inference pipeline for the engine sound classification system.
It loads the SVM model and YAMNet to predict the mechanical health status of an engine from audio.
"""

import os
import sys
import time
import json
import pathlib
import numpy as np
import librosa
import joblib
import tensorflow as tf
import tensorflow_hub as hub

# Configuration variables
MODEL_PATH = "models/saved/svm_model.joblib"
SCALER_PATH = "models/saved/scaler.joblib"
LABEL_MAP_PATH = "data/processed/embeddings.json"
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16000
MIN_DURATION = 3.0
MAX_DURATION = float(os.getenv("MAX_DURATION", 120.0))
MIN_AMPLITUDE = 0.01

# --- Content validation (YAMNet AudioSet head) ---
# YAMNet's pretrained scores let us verify the clip is actually an engine.
# These display names come from yamnet_class_map.csv and are resolved to
# indices once, lazily, in _get_yamnet_classes().
ENGINE_PRESENCE_CLASSES = [
    "Engine", "Vehicle", "Car",
    "Medium engine (mid frequency)",
    "Light engine (high frequency)",
    "Heavy engine (low frequency)",
]
STAGE_CLASSES = {
    "audio_start": "Engine starting",
    "audio_idle": "Idling",
    "audio_acceleration": "Accelerating, revving, vroom",
}
# Calibrated on labelled clips + synthetic negatives:
#   silence / tones / white / pink noise -> 0.00-0.03   (reject)
#   faulty battery-crank start (real engine) -> 0.086    (keep)
#   real running engines -> 0.25-1.33                    (keep)
# 0.04 sits between garbage and the faulty-engine floor.
# LIMITATION: speech/music can score ~0.15 (above a faulty crank), so this
# gate catches obvious non-engine noise but NOT speech/music. Raising the
# threshold to catch speech would also reject faulty starts. Recalibrate with
# more faulty-engine + true-negative samples before tightening.
ENGINE_GATE_MIN = float(os.getenv("ENGINE_GATE_MIN", 0.04))
# Normal starts score >= ~0.75 on "Engine starting"; steady-state clips peak
# at ~0.48. 0.5 cleanly separates a cranking transient from a running engine.
# NOTE: idle vs high-RPM are NOT separable with YAMNet's head, so stage
# checks are advisory (soft warnings), never a hard block.
STAGE_START_HINT = float(os.getenv("STAGE_START_HINT", 0.5))
STAGE_STEADY_HINT = float(os.getenv("STAGE_STEADY_HINT", 0.3))

# Lazily-resolved {class_name: index} cache, keyed on the yamnet model id.
_YAMNET_CLASS_IDX = None

# MHS scoring constants
BASE_SCORES = {
    "healthy": 100,
    "knocking": 55,
    "misfiring": 50,
    "rotational_imbalance": 60,
    "tappet": 65,
    "battery_fault": 70
}
PENALTY_WEIGHT = 30

# Plain English explanations
FAULT_EXPLANATIONS = {
    "healthy": "The engine sounds healthy. No faults detected.",
    "knocking": "Engine knocking detected. This is a serious fault that may indicate low oil pressure or fuel quality issues. Have this inspected before purchasing.",
    "misfiring": "Engine misfiring detected. The engine is not firing consistently. This could indicate spark plug or fuel injector issues.",
    "rotational_imbalance": "Rotational imbalance detected. A component may be worn or loose, causing uneven rotation.",
    "tappet": "Tappet noise detected. This clicking sound may indicate valve clearance issues, often worse when the engine is cold.",
    "battery_fault": "Battery or starting fault detected. The engine struggled to start normally. Battery or starter motor may need attention."
}


def load_models():
    """
    Load the SVM model, scaler, label map, and YAMNet model.
    Raises FileNotFoundError if any critical file is missing.
    Returns: (yamnet_model, svm_model, scaler, label_map)
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file missing: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file missing: {SCALER_PATH}")
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map missing: {LABEL_MAP_PATH}")

    print("Loading SVM model...")
    svm_model = joblib.load(MODEL_PATH)
    print("SVM loaded.")

    print("Loading Scaler...")
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded.")

    print("Loading Label Map...")
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    print("Label map loaded.")

    print("Loading YAMNet...")
    yamnet_model = hub.load(YAMNET_URL)
    print("YAMNet loaded.")

    return yamnet_model, svm_model, scaler, label_map


def validate_audio(file_path):
    """
    Load the audio file to validate duration and amplitude.
    Returns: dict with keys: is_valid (bool), duration (float), reason (str).
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)
        duration = float(librosa.get_duration(y=audio, sr=sr))
        
        if duration < MIN_DURATION:
            return {"is_valid": False, "code": "TOO_SHORT", "duration": duration,
                    "reason": f"Recording is too short ({duration:.1f}s). Please record at least {MIN_DURATION:.0f}s."}

        if duration > MAX_DURATION:
            return {"is_valid": False, "code": "TOO_LONG", "duration": duration,
                    "reason": f"Recording is too long ({duration:.0f}s). Please keep it under {MAX_DURATION:.0f}s."}

        max_amplitude = float(np.max(np.abs(audio)))
        if max_amplitude < MIN_AMPLITUDE:
            return {"is_valid": False, "code": "TOO_QUIET", "duration": duration,
                    "reason": "Recording is too quiet. Please record closer to the engine (~10 cm)."}

        return {"is_valid": True, "code": "OK", "duration": duration, "reason": "OK"}

    except Exception as e:
        return {"is_valid": False, "code": "UNREADABLE", "duration": 0.0,
                "reason": "This audio file could not be read. Please re-record or pick another file."}


def _get_yamnet_classes(yamnet_model):
    """Resolve and cache the {display_name: index} map from YAMNet's class map CSV."""
    global _YAMNET_CLASS_IDX
    if _YAMNET_CLASS_IDX is not None:
        return _YAMNET_CLASS_IDX

    import csv
    class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
    names = []
    with open(class_map_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            names.append(row[2])  # display_name column

    wanted = set(ENGINE_PRESENCE_CLASSES) | set(STAGE_CLASSES.values())
    _YAMNET_CLASS_IDX = {name: i for i, name in enumerate(names) if name in wanted}
    return _YAMNET_CLASS_IDX


def run_yamnet(yamnet_model, file_path):
    """
    Single YAMNet pass. Returns dict with the mean embedding (for the SVM),
    plus per-class mean/max scores (for content validation), or None on failure.
    """
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)

        scores, embeddings, spectrogram = yamnet_model(audio)
        scores = scores.numpy()  # (frames, 521)

        return {
            "embedding": np.mean(embeddings.numpy(), axis=0),
            "scores_mean": scores.mean(axis=0),
            "scores_max": scores.max(axis=0),
        }
    except Exception as e:
        print(f"YAMNet pass failed: {str(e)}")
        return None


def check_content(scores_mean, scores_max, expected_stage, class_idx):
    """
    Content validation from YAMNet scores.
      - is_engine (HARD): summed engine-presence classes must clear the gate.
      - stage warning (SOFT): flags an obvious start/steady mismatch only;
        idle vs high-RPM is intentionally NOT distinguished (not separable).
    Returns: {is_engine, engine_score, warning (str|None), detected}
    """
    engine_score = float(sum(
        scores_mean[class_idx[c]] for c in ENGINE_PRESENCE_CLASSES if c in class_idx
    ))
    if engine_score < ENGINE_GATE_MIN:
        return {"is_engine": False, "engine_score": engine_score,
                "warning": None, "detected": None}

    start_score = float(scores_max[class_idx[STAGE_CLASSES["audio_start"]]]) \
        if STAGE_CLASSES["audio_start"] in class_idx else 0.0
    idle_score = float(scores_max[class_idx[STAGE_CLASSES["audio_idle"]]]) \
        if STAGE_CLASSES["audio_idle"] in class_idx else 0.0
    accel_score = float(scores_max[class_idx[STAGE_CLASSES["audio_acceleration"]]]) \
        if STAGE_CLASSES["audio_acceleration"] in class_idx else 0.0
    steady_score = max(idle_score, accel_score)

    looks_like_start = start_score >= STAGE_START_HINT
    detected = "start" if looks_like_start else "running_engine"

    warning = None
    if expected_stage == "audio_start":
        if not looks_like_start and steady_score >= STAGE_STEADY_HINT:
            warning = ("This sounds like a running engine rather than a cold start. "
                       "Record from the moment you turn the ignition.")
    elif expected_stage in ("audio_idle", "audio_acceleration"):
        if looks_like_start:
            warning = ("This sounds like an engine start rather than a steady "
                       "idle/acceleration clip. Check you're on the right stage.")

    return {"is_engine": True, "engine_score": engine_score,
            "warning": warning, "detected": detected}


def compute_mhs(fault_class, confidence):

    if fault_class == "healthy":

        mhs = int(round(confidence * 100))
    else:

        if fault_class not in BASE_SCORES:
            return 0
        base = BASE_SCORES[fault_class]
        mhs = int(round(base - (PENALTY_WEIGHT * confidence)))
    
    return max(0, min(100, mhs))


def get_color_indicator(mhs_score):
    """
    Return green for >= 80, amber for >= 50, red for < 50.
    """
    if mhs_score >= 80:
        return "green"
    elif mhs_score >= 50:
        return "amber"
    else:
        return "red"


def _health_from_embedding(embedding, svm_model, scaler, label_map, duration):
    """Run the SVM health classifier on a YAMNet embedding and build the result dict."""
    embedding_scaled = scaler.transform([embedding])
    probs = svm_model.predict_proba(embedding_scaled)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(np.max(probs))

    fault_class = "unknown"
    for name, idx in label_map.items():
        if idx == pred_idx:
            fault_class = name
            break

    mhs_score = compute_mhs(fault_class, confidence)
    return {
        "status": "success",
        "fault_class": fault_class,
        "confidence": round(confidence, 4),
        "confidence_percent": f"{int(round(confidence * 100))}%",
        "mhs_score": mhs_score,
        "color_indicator": get_color_indicator(mhs_score),
        "explanation": FAULT_EXPLANATIONS.get(fault_class, "Unknown fault."),
        "recommend_professional": mhs_score < 70,
        "duration_seconds": round(duration, 2),
    }


def validate_and_analyze_stage(file_path, expected_stage, yamnet_model, svm_model, scaler, label_map):
    """
    Full per-stage pipeline for one file:
      1. Quality (HARD): decodable, duration in range, not silent.
      2. Engine gate (HARD): YAMNet must recognise it as an engine.
      3. Stage identity (SOFT): advisory warning on start/steady mismatch.
      4. Health prediction (reuses the YAMNet embedding from step 2/3).
    Returns a per-stage dict; 'valid' is False if a HARD check failed.
    """
    val = validate_audio(file_path)
    if not val["is_valid"]:
        return {"valid": False, "code": val["code"], "message": val["reason"]}

    yam = run_yamnet(yamnet_model, file_path)
    if yam is None:
        return {"valid": False, "code": "FEATURE_FAIL", "duration": round(val["duration"], 2),
                "message": "Could not analyse this recording. Please re-record it."}

    class_idx = _get_yamnet_classes(yamnet_model)
    content = check_content(yam["scores_mean"], yam["scores_max"], expected_stage, class_idx)
    if not content["is_engine"]:
        return {"valid": False, "code": "NOT_ENGINE", "duration": round(val["duration"], 2),
                "engine_score": round(content["engine_score"], 3),
                "message": ("This doesn't sound like an engine. Record the running engine "
                            "with the bonnet open, about 10 cm away.")}

    stage = {"valid": True, "code": "OK", "duration": round(val["duration"], 2),
             "detected": content["detected"]}
    stage.update(_health_from_embedding(yam["embedding"], svm_model, scaler, label_map, val["duration"]))
    if content["warning"]:
        stage["warning"] = content["warning"]
    return stage


# Fields safe to expose to the client (drops internal health scoring detail).
_PUBLIC_STAGE_KEYS = ("valid", "code", "message", "warning", "duration", "detected", "mhs_score")


def _public_stages(stages):
    """Compact per-stage summary for the API response / mobile client."""
    return {
        field: {k: s[k] for k in _PUBLIC_STAGE_KEYS if k in s}
        for field, s in stages.items()
    }


def analyze(file_map, yamnet_model, svm_model, scaler, label_map):
    """
    Validate and analyse one or more stage recordings.
    file_map: ordered {field_name: file_path}, where field_name is one of
    'audio_start' / 'audio_idle' / 'audio_acceleration' (or any label for a
    legacy single file).

    HARD failures (bad audio or not-an-engine) in ANY stage block the whole
    analysis and return status 'validation_error' with per-stage messages.
    Otherwise returns worst-case ('lowest MHS') health plus per-stage warnings.
    """
    stages = {}
    for field, path in file_map.items():
        expected = field if field in STAGE_CLASSES else None
        stages[field] = validate_and_analyze_stage(
            path, expected, yamnet_model, svm_model, scaler, label_map
        )

    if any(not s["valid"] for s in stages.values()):
        return {
            "status": "validation_error",
            "message": "Some recordings couldn't be validated. Please re-record the highlighted stages.",
            "stages": _public_stages(stages),
        }

    # All stages valid: worst-case aggregation (a fault in any stage matters).
    worst = min(stages.values(), key=lambda s: s["mhs_score"])
    result = {
        "status": "success",
        "fault_class": worst["fault_class"],
        "confidence": worst["confidence"],
        "confidence_percent": worst["confidence_percent"],
        "mhs_score": worst["mhs_score"],
        "color_indicator": worst["color_indicator"],
        "explanation": worst["explanation"],
        "recommend_professional": worst["recommend_professional"],
        "is_multi_stage": len(file_map) > 1,
        "total_stages": len(file_map),
        "stages_passed": len(stages),
        "stages": _public_stages(stages),
    }
    return result


def predict(file_path, yamnet_model, svm_model, scaler, label_map, expected_stage=None):
    """Single-file convenience wrapper (used by the /test endpoint and CLI)."""
    stage = validate_and_analyze_stage(
        file_path, expected_stage, yamnet_model, svm_model, scaler, label_map
    )
    if not stage["valid"]:
        return {"status": "unclassified", "mhs_score": 0,
                "code": stage.get("code"), "reason": stage["message"]}
    result = {k: v for k, v in stage.items() if k not in ("valid", "code")}
    result["file_path"] = file_path
    return result


def predict_multi(file_paths, yamnet_model, svm_model, scaler, label_map):
    """Backward-compatible wrapper around analyze() for positional file lists."""
    file_map = {f"audio_{i}": p for i, p in enumerate(file_paths) if os.path.exists(p)}
    if not file_map:
        return {"status": "unclassified", "mhs_score": 0,
                "reason": "No valid audio files provided for multi-stage analysis."}
    return analyze(file_map, yamnet_model, svm_model, scaler, label_map)


if __name__ == "__main__":

    try:
        yamnet, svm, scaler, label_map = load_models()
    except Exception as e:
        print(f"Initialization error: {e}")
        sys.exit(1)

    # Determine files to test
    test_files = []
    
    if len(sys.argv) > 1:
        test_files.append(sys.argv[1])
    else:
        # Default test cases
        healthy_dir = pathlib.Path("data/test/healthy")
        if healthy_dir.exists():
            wavs = list(healthy_dir.glob("*.wav"))
            if wavs:
                test_files.append(str(wavs[0]))
                
        knocking_dir = pathlib.Path("data/test/knocking")
        if knocking_dir.exists():
            wavs = list(knocking_dir.glob("*.wav"))
            if wavs:
                test_files.append(str(wavs[0]))
                
    if not test_files:
        print("No test files provided or found in data/test/.")
        sys.exit(1)
        
    for fpath in test_files:
        print(f"\n--- Testing file: {fpath} ---")
        start_time = time.time()
        
        result = predict(fpath, yamnet, svm, scaler, label_map)
        
        end_time = time.time()
        inference_time_ms = int((end_time - start_time) * 1000)
        
        # Print cleanly
        print(json.dumps(result, indent=4))
        print(f"Total inference time: {inference_time_ms} ms")

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
# A healthy engine can catch in under two seconds, so the start stage gets a
# shorter minimum (real starts in data/test are 1.6-2.2 s long).
MIN_DURATION_START = float(os.getenv("MIN_DURATION_START", 1.5))

# --- Single guided recording ---
# The app asks for one ~30 s recording: start the engine, let it idle, then
# rev gently, with on-screen prompts. The server cuts it into the same three
# stages at the times the app showed (sent with the upload), falling back to
# these defaults. Idle and revving are cut by time, not detected, because
# YAMNet cannot tell them apart reliably.
MIN_DURATION_FULL = float(os.getenv("MIN_DURATION_FULL", 20.0))
DEFAULT_START_END = 8.0
DEFAULT_IDLE_END = 20.0
MIN_SEGMENT = 2.0          # shortest stage we will classify
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
# 0.04 sits between garbage and the faulty-engine floor. Raising it would
# reject faulty starts, so sounds that clear it without being an engine are
# caught by the competing-class check below instead.
ENGINE_GATE_MIN = float(os.getenv("ENGINE_GATE_MIN", 0.04))
# Competing sounds: a clip is rejected when any of these groups scores at
# least as high as the engine-presence classes. Calibrated on the 16 real
# clips in data/test (all kept; worst case is the faulty crank, engine 0.086
# vs speech 0.051) and on speech, music, birdsong, whale song, tones, hum and
# white/pink noise (all rejected, including white noise that scores 0.079 on
# the engine classes and so passes ENGINE_GATE_MIN on its own). Talking
# quietly over an idling engine still passes; talking at a similar level to
# the engine does not.
COMPETING_CLASSES = {
    "speech": ["Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue",
               "Male speech, man speaking", "Female speech, woman speaking"],
    "music": ["Music", "Musical instrument", "Singing"],
    "background noise": ["White noise", "Pink noise", "Static"],
    "animal sounds": ["Animal", "Wild animals", "Bird", "Domestic animals, pets", "Dog", "Cat"],
}
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

# Plain English explanations: state the issue detected and whether a mechanic
# inspection is advised — without disclosing the underlying cause of the fault.
FAULT_EXPLANATIONS = {
    "healthy": "The engine sounds healthy. No faults detected — a further mechanical inspection isn't needed based on the audio.",
    "knocking": "Engine knocking detected. We recommend having a mechanic inspect the engine before you buy.",
    "misfiring": "Engine misfiring detected. We recommend having a mechanic inspect the engine before you buy.",
    "rotational_imbalance": "Rotational imbalance detected. We recommend having a mechanic inspect the engine before you buy.",
    "tappet": "Tappet noise detected. We recommend having a mechanic inspect the engine before you buy.",
    "battery_fault": "Battery or starting fault detected. We recommend having a mechanic inspect the engine before you buy."
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


def validate_audio(file_path, expected_stage=None, min_duration=None):
    """
    Load the audio file to validate duration and amplitude.
    Returns: dict with keys: is_valid (bool), duration (float), reason (str).
    """
    if min_duration is None:
        min_duration = MIN_DURATION_START if expected_stage == "audio_start" else MIN_DURATION
    try:
        audio, sr = librosa.load(file_path, sr=None)
        duration = float(librosa.get_duration(y=audio, sr=sr))

        if duration < min_duration:
            return {"is_valid": False, "code": "TOO_SHORT", "duration": duration,
                    "reason": f"Recording is too short ({duration:.1f}s). Please record at least {min_duration:g}s."}

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
    wanted |= {c for group in COMPETING_CLASSES.values() for c in group}
    _YAMNET_CLASS_IDX = {name: i for i, name in enumerate(names) if name in wanted}
    return _YAMNET_CLASS_IDX


# A fixed, inaudible noise floor (about -80 dBFS) added before YAMNet. 16-bit
# recordings turn near-silent stretches (e.g. before the key is turned) into
# runs of one-step values, and the classifier reacted to that pattern: a
# healthy start was rated "misfiring" after a 16-bit save. With this floor the
# 16 clips in data/test keep their classes and the 16-bit case is stable.
# Seeded so the same file always gives the same result.
DITHER_LEVEL = 1e-4


def _prepare(audio):
    audio = np.asarray(audio, dtype=np.float32)
    noise = np.random.default_rng(0).standard_normal(len(audio)).astype(np.float32)
    return audio + DITHER_LEVEL * noise


def run_yamnet(yamnet_model, file_path):
    """
    Single YAMNet pass. Returns dict with the mean embedding (for the SVM),
    plus per-class mean/max scores (for content validation), or None on failure.
    """
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        scores, embeddings, spectrogram = yamnet_model(_prepare(audio))
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
      - competing sound (HARD): speech, music, noise or animal sounds must not
        score as high as the engine.
      - stage warning (SOFT): flags an obvious start/steady mismatch only;
        idle vs high-RPM is intentionally NOT distinguished (not separable).
    Returns: {is_engine, engine_score, competing (str|None), warning (str|None), detected}
    """
    engine_score = float(sum(
        scores_mean[class_idx[c]] for c in ENGINE_PRESENCE_CLASSES if c in class_idx
    ))
    if engine_score < ENGINE_GATE_MIN:
        return {"is_engine": False, "engine_score": engine_score, "competing": None,
                "warning": None, "detected": None}

    competing = {
        group: float(sum(scores_mean[class_idx[c]] for c in classes if c in class_idx))
        for group, classes in COMPETING_CLASSES.items()
    }
    loudest = max(competing, key=competing.get)
    if competing[loudest] >= engine_score:
        return {"is_engine": False, "engine_score": engine_score, "competing": loudest,
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

    return {"is_engine": True, "engine_score": engine_score, "competing": None,
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
    probs = svm_model.predict_proba(scaler.transform([embedding]))[0]
    return _health_from_probs(probs, label_map, duration)


# Idle and revving stages of a single recording are scored in ~4 s windows
# (YAMNet frames hop 0.48 s) and the SVM probabilities averaged. Averaging the
# embedding over the whole 12 s stage instead mixes two engine states whenever
# the driver is a few seconds off the prompts (start tail + idle, or idle +
# first revs); the classifier was trained on single-state clips and read those
# mixes as "knocking", turning healthy cars red. Continuous faults still win
# every window.
STAGE_WINDOW_FRAMES = 7
STAGE_WINDOW_HOP = 4


def _windowed_probs(embeddings, svm_model, scaler):
    """Mean SVM probabilities over overlapping windows of YAMNet frame embeddings."""
    starts = range(0, max(len(embeddings) - STAGE_WINDOW_FRAMES, 0) + 1, STAGE_WINDOW_HOP)
    pooled = np.array([embeddings[s:s + STAGE_WINDOW_FRAMES].mean(axis=0) for s in starts])
    return svm_model.predict_proba(scaler.transform(pooled)).mean(axis=0)


def _health_from_probs(probs, label_map, duration):
    """Build the health result dict from SVM class probabilities."""
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
        "explanation": FAULT_EXPLANATIONS.get(fault_class, "An unusual engine sound was detected. We recommend having a mechanic inspect the engine before you buy."),
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
    val = validate_audio(file_path, expected_stage)
    if not val["is_valid"]:
        return {"valid": False, "code": val["code"], "message": val["reason"]}

    yam = run_yamnet(yamnet_model, file_path)
    if yam is None:
        return {"valid": False, "code": "FEATURE_FAIL", "duration": round(val["duration"], 2),
                "message": "Could not analyse this recording. Please re-record it."}

    class_idx = _get_yamnet_classes(yamnet_model)
    content = check_content(yam["scores_mean"], yam["scores_max"], expected_stage, class_idx)
    if content["competing"]:
        return {"valid": False, "code": "OTHER_SOUND", "duration": round(val["duration"], 2),
                "engine_score": round(content["engine_score"], 3),
                "message": (f"We could hear {content['competing']} more clearly than the engine. "
                            "Record again close to the engine, without talking or music nearby.")}
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
_PUBLIC_STAGE_KEYS = ("valid", "code", "message", "warning", "duration", "detected", "mhs_score",
                      "start", "end")


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

    return _summarise(stages)


def _summarise(stages):
    """Worst-case aggregation over valid stages (a fault in any stage matters)."""
    worst = min(stages.values(), key=lambda s: s["mhs_score"])
    return {
        "status": "success",
        "fault_class": worst["fault_class"],
        "confidence": worst["confidence"],
        "confidence_percent": worst["confidence_percent"],
        "mhs_score": worst["mhs_score"],
        "color_indicator": worst["color_indicator"],
        "explanation": worst["explanation"],
        "recommend_professional": worst["recommend_professional"],
        "is_multi_stage": len(stages) > 1,
        "total_stages": len(stages),
        "stages_passed": len(stages),
        "stages": _public_stages(stages),
    }


def _stage_bounds(duration, start_end=None, idle_end=None):
    """
    Start/idle/acceleration boundaries in seconds, clamped so every stage is
    at least MIN_SEGMENT long.
    """
    start_end = DEFAULT_START_END if start_end is None else float(start_end)
    idle_end = DEFAULT_IDLE_END if idle_end is None else float(idle_end)
    start_end = min(max(start_end, MIN_SEGMENT), duration - 2 * MIN_SEGMENT)
    idle_end = min(max(idle_end, start_end + MIN_SEGMENT), duration - MIN_SEGMENT)
    return {
        "audio_start": (0.0, start_end),
        "audio_idle": (start_end, idle_end),
        "audio_acceleration": (idle_end, duration),
    }


def analyze_recording(file_path, yamnet_model, svm_model, scaler, label_map,
                      start_end=None, idle_end=None):
    """
    Analyse one continuous recording that covers start, idle and revving.

    The whole clip is checked once (readable, long enough, an engine, not
    drowned out), then cut into the three stages. Each stage is analysed on
    its own, as if it had been uploaded separately, and the worst stage
    decides the result. If no engine start is heard, the start stage still gets a
    result but carries a warning, because battery and starter faults can only
    be heard at start.
    """
    def rejected(code, message, duration=None):
        stage = {"valid": False, "code": code, "message": message}
        if duration is not None:
            stage["duration"] = round(duration, 2)
        return {"status": "validation_error", "message": message, "stages": {"audio_full": stage}}

    val = validate_audio(file_path, min_duration=MIN_DURATION_FULL)
    if not val["is_valid"]:
        return rejected(val["code"], val["reason"], val["duration"])
    duration = val["duration"]

    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        scores = yamnet_model(_prepare(audio))[0].numpy()
    except Exception as e:
        print(f"YAMNet pass failed: {str(e)}")
        return rejected("FEATURE_FAIL", "Could not analyse this recording. Please re-record it.", duration)

    class_idx = _get_yamnet_classes(yamnet_model)
    content = check_content(scores.mean(axis=0), scores.max(axis=0), None, class_idx)
    if content["competing"]:
        return rejected("OTHER_SOUND",
                        f"We could hear {content['competing']} more clearly than the engine. "
                        "Record again close to the engine, without talking or music nearby.", duration)
    if not content["is_engine"]:
        return rejected("NOT_ENGINE",
                        "This doesn't sound like an engine. Record the running engine "
                        "with the bonnet open, about 10 cm away.", duration)

    starting_idx = class_idx.get(STAGE_CLASSES["audio_start"])
    stages = {}
    for field, (begin, end) in _stage_bounds(duration, start_end, idle_end).items():
        # Each stage is cut out and run through YAMNet on its own, exactly like
        # a separately uploaded clip, which is what the classifier was trained on.
        segment = audio[int(begin * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        seg_scores, seg_embeddings, _ = yamnet_model(_prepare(segment))
        seg_scores = seg_scores.numpy()
        seg_embeddings = seg_embeddings.numpy()
        stage = {"valid": True, "code": "OK", "duration": round(end - begin, 2),
                 "start": round(begin, 2), "end": round(end, 2)}
        if field == "audio_start":
            # Kept whole: a start is one short event, and battery faults show in it as a whole.
            stage.update(_health_from_embedding(seg_embeddings.mean(axis=0), svm_model, scaler,
                                                label_map, end - begin))
        else:
            stage.update(_health_from_probs(_windowed_probs(seg_embeddings, svm_model, scaler),
                                            label_map, end - begin))
        if field == "audio_start":
            heard_start = starting_idx is not None and seg_scores[:, starting_idx].max() >= STAGE_START_HINT
            stage["detected"] = "start" if heard_start else "running_engine"
            if not heard_start:
                stage["warning"] = ("We didn't hear the engine start, so battery and starter problems "
                                    "couldn't be checked. Start recording before you turn the key.")
        else:
            stage["detected"] = "running_engine"
        stages[field] = stage

    result = _summarise(stages)
    result["mode"] = "single_recording"
    result["duration"] = round(duration, 2)
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

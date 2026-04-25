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
MIN_AMPLITUDE = 0.01

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
            return {"is_valid": False, "duration": duration, "reason": f"Audio is too short ({duration:.2f}s). Minimum {MIN_DURATION}s required."}
            
        max_amplitude = float(np.max(np.abs(audio)))
        if max_amplitude < MIN_AMPLITUDE:
            return {"is_valid": False, "duration": duration, "reason": "Audio is too quiet. Please record closer to the engine."}
            
        return {"is_valid": True, "duration": duration, "reason": "OK"}
        
    except Exception as e:
        return {"is_valid": False, "duration": 0.0, "reason": f"Failed to load audio: {str(e)}"}


def extract_embedding(yamnet_model, file_path):
    """
    Load audio at 16000Hz, ensure mono/float32, and extract YAMNet embedding.
    Returns: (1024,) numpy array or None if fails.
    """
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)
        
        # Pass through YAMNet to get scores, embeddings, spectrogram
        scores, embeddings, spectrogram = yamnet_model(audio)
        
        # Take mean across frames to get single vector
        embedding_mean = np.mean(embeddings.numpy(), axis=0)
        return embedding_mean
        
    except Exception as e:
        print(f"Extraction failed: {str(e)}")
        return None


def compute_mhs(fault_class, confidence):
    """
    Calculate MHS based on fault class and model confidence.
    For healthy class, higher confidence leads to a higher score.
    For fault classes, the score starts at a base and is penalized by confidence.
    """
    if fault_class == "healthy":
        # Higher confidence = higher score
        # confidence 1.0 → MHS 100
        # confidence 0.5 → MHS 50
        mhs = int(round(confidence * 100))
    else:
        # Fault detected — apply penalty
        # BASE_SCORE is the ceiling for that fault
        # Higher confidence in fault = lower score
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


def predict(file_path, yamnet_model, svm_model, scaler, label_map):
    """
    Main prediction pipeline.
    Validates audio, extracts embedding, predicts, calculates MHS, and returns result dict.
    """
    val_result = validate_audio(file_path)
    if not val_result["is_valid"]:
        return {
            "status": "unclassified",
            "mhs_score": 0,
            "reason": val_result["reason"]
        }
        
    embedding = extract_embedding(yamnet_model, file_path)
    if embedding is None:
        return {
            "status": "unclassified",
            "mhs_score": 0,
            "reason": "Failed to extract features from audio."
        }
        
    # Scale the embedding
    embedding_scaled = scaler.transform([embedding])
    
    # Get SVM prediction
    pred_idx = svm_model.predict(embedding_scaled)[0]
    probs = svm_model.predict_proba(embedding_scaled)[0]
    confidence = float(np.max(probs))
    
    # Reverse lookup for class name
    fault_class = "unknown"
    for name, idx in label_map.items():
        if idx == pred_idx:
            fault_class = name
            break
            
    mhs_score = compute_mhs(fault_class, confidence)
    color_indicator = get_color_indicator(mhs_score)
    explanation = FAULT_EXPLANATIONS.get(fault_class, "Unknown fault.")
    
    return {
        "status": "success",
        "fault_class": fault_class,
        "confidence": round(confidence, 4),
        "confidence_percent": f"{int(round(confidence * 100))}%",
        "mhs_score": mhs_score,
        "color_indicator": color_indicator,
        "explanation": explanation,
        "recommend_professional": mhs_score < 70,
        "file_path": file_path,
        "duration_seconds": round(val_result["duration"], 2)
    }


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

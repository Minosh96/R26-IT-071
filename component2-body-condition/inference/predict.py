import os
import io
import uuid
from PIL import Image
from ultralytics import YOLO

CLASS_PENALTY = {
    "Dent": 15,
    "Scratch": 10,
    "Rust": 12,
}

def load_models():
    # Make sure we load the damage_model.pt
    MODEL_PATH = os.getenv("MODEL_PATH", "damage_model.pt")
    
    # Check if the path is relative to the root or relative to the inference dir
    if not os.path.exists(MODEL_PATH):
        # Try to find it in the current working directory, or parent directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt_path = os.path.join(parent_dir, MODEL_PATH)
        if os.path.exists(alt_path):
            MODEL_PATH = alt_path
        else:
            raise FileNotFoundError(f"YOLO model not found at {MODEL_PATH} or {alt_path}")

    print(f"Loading YOLO model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    return model

def calculate_score(detections):
    score = 100.0

    for det in detections:
        label = det.get("label", "")
        confidence = float(det.get("confidence", 0))

        if label in CLASS_PENALTY:
            penalty = CLASS_PENALTY[label] * confidence
            score -= penalty

    return max(0, min(100, round(score, 2)))

def get_severity(score):
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Moderate"
    elif score >= 30:
        return "Poor"
    else:
        return "Severe"

def predict_body_condition(model, views, conf_threshold=0.25):
    """
    Analyzes multiple views of a vehicle.
    `views` is a dictionary of {view_name: bytes}
    """
    session_id = str(uuid.uuid4())
    view_analysis = {}
    all_detections = []
    
    # Ensure directories exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    for view_name, contents in views.items():
        if not contents:
            raise ValueError(f"The {view_name} image is empty.")

        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            raise ValueError(f"The {view_name} file is not a valid image.")

        upload_path = os.path.join("uploads", f"{session_id}_{view_name}.jpg")
        image.save(upload_path)

        results = model.predict(
            source=image,
            conf=conf_threshold,
            save=False
        )

        view_detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                detection = {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "bbox": [round(x, 2) for x in bbox]
                }

                view_detections.append(detection)
                all_detections.append(detection)

        # Generate visualized result
        result_image_bgr = results[0].plot()
        result_image_rgb = result_image_bgr[:, :, ::-1] # Convert BGR to RGB
        result_image = Image.fromarray(result_image_rgb)
        
        output_filename = f"result_{session_id}_{view_name}.jpg"
        output_path = os.path.join("outputs", output_filename)
        result_image.save(output_path)

        view_analysis[view_name] = {
            "count": len(view_detections),
            "issues": view_detections,
            "visual_url": f"/outputs/{output_filename}"
        }

    condition_score = calculate_score(all_detections)
    severity = get_severity(condition_score)

    return {
        "session_id": session_id,
        "condition_score": condition_score,
        "severity": severity,
        "total_detection_count": len(all_detections),
        "view_analysis": view_analysis,
        "summary": f"Analysis complete. Found {len(all_detections)} damages across {len(views)} views. Score: {condition_score} ({severity})."
    }

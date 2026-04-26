import os
import io
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Vehicle Body Condition Analysis API")

for folder in ["uploads", "outputs", "runs"]:
    os.makedirs(folder, exist_ok=True)

MODEL_PATH = os.getenv("MODEL_PATH", "damage_model.pt")
CONF_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.25))

model = None


@app.on_event("startup")
async def startup_event():
    global model

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("Put damage_model.pt in the same folder as main.py")
        model = None
        return

    print(f"Loading model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")
    print("Loaded model classes:", model.names)


CLASS_PENALTY = {
    "Bonnet": 6,
    "Bumper": 8,
    "Dent": 15,
    "Dickey": 6,
    "Door": 6,
    "Fender": 6,
    "Light": 5,
    "Scratch": 10,
    "Windshield": 5,
}


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


@app.get("/")
async def root():
    return {
        "message": "Vehicle Damage Detection Backend is running.",
        "docs": "Open http://127.0.0.1:8000/docs",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_classes": model.names if model is not None else None,
    }


@app.post("/analyze")
async def analyze_vehicle(
    front: UploadFile = File(...),
    rear: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    roof: UploadFile = File(...)
):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Make sure damage_model.pt is in the same folder as main.py."
        )

    views = {
        "front": front,
        "rear": rear,
        "left": left,
        "right": right,
        "roof": roof
    }

    session_id = str(uuid.uuid4())
    view_analysis = {}
    all_detections = []

    try:
        for view_name, file in views.items():
            contents = await file.read()

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
                conf=CONF_THRESHOLD,
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

            view_analysis[view_name] = {
                "count": len(view_detections),
                "issues": view_detections
            }

        condition_score = calculate_score(all_detections)
        severity = get_severity(condition_score)

        return {
            "session_id": session_id,
            "condition_score": condition_score,
            "severity": severity,
            "total_detection_count": len(all_detections),
            "view_analysis": view_analysis,
            "summary": f"Analysis complete. Found {len(all_detections)} damages across 5 views. Score: {condition_score} ({severity})."
        }

    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
import os
import io
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image


app = FastAPI(
    title="Vehicle Damage Type Classification API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_PATH = "vehicle_damage_type_mobilenetv3_best.h5"
IMG_SIZE = (224, 224)

# This must match your training output:
# Class names: ['dent', 'rust', 'scratch', 'undamaged']
CLASS_NAMES = ["dent", "rust", "scratch", "undamaged"]

VIEW_WEIGHTS = {
    "front": 0.25,
    "rear": 0.25,
    "left": 0.20,
    "right": 0.20,
    "up": 0.10,
}

DAMAGE_PENALTIES = {
    "undamaged": 0,
    "scratch": 10,
    "dent": 20,
    "rust": 25,
}

damage_type_model = None


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------

@app.on_event("startup")
def load_model():
    global damage_type_model

    print("\n==========================================")
    print("STARTING VEHICLE DAMAGE TYPE API v2.0.0")
    print("==========================================")
    print("Current folder:", os.getcwd())
    print("Expected model:", MODEL_PATH)
    print("Model exists:", os.path.exists(MODEL_PATH))

    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model file '{MODEL_PATH}' not found.")
        print("[WARNING] Copy vehicle_damage_type_mobilenetv3_best.h5 into this backend folder.")
        damage_type_model = None
        return

    try:
        print(f"[INFO] Loading model from: {MODEL_PATH}")
        damage_type_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("[INFO] Damage type model loaded successfully.")
        print("[INFO] Class order:", CLASS_NAMES)

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        damage_type_model = None


# ---------------------------------------------------------
# Debug endpoints
# ---------------------------------------------------------

@app.get("/version")
def version():
    return {
        "api_version": "2.0.0",
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "class_names": CLASS_NAMES,
        "message": "This backend uses dent/rust/scratch/undamaged MobileNetV3 model"
    }


@app.get("/routes")
def routes():
    return {
        "routes": [
            route.path
            for route in app.routes
        ]
    }


# ---------------------------------------------------------
# Home page
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def read_index():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()

    return """
    <html>
        <body>
            <h2>Vehicle Damage Type Classification API is running</h2>
            <p>Go to <a href="/docs">Swagger Docs</a> to test the API.</p>
        </body>
    </html>
    """


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def preprocess_image(uploaded_file: UploadFile):
    try:
        uploaded_file.file.seek(0)
        image_bytes = uploaded_file.file.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)

        image_array = np.array(image)

        # Do NOT divide by 255.
        # Your MobileNetV3 model was trained with include_preprocessing=True.
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {uploaded_file.filename}. Error: {str(e)}"
        )


def classify_damage_type(uploaded_file: UploadFile, view_name: str):
    global damage_type_model

    if damage_type_model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Please place '{MODEL_PATH}' inside the backend folder."
        )

    image_array = preprocess_image(uploaded_file)

    predictions = damage_type_model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    probabilities = {
        CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    damaged_probability = round(
        probabilities["dent"] +
        probabilities["rust"] +
        probabilities["scratch"],
        2
    )

    undamaged_probability = probabilities["undamaged"]

    if predicted_class == "undamaged":
        status = "undamaged"
        damage_type = None
    else:
        status = "damaged"
        damage_type = predicted_class

    penalty = DAMAGE_PENALTIES[predicted_class]
    view_score = max(0, 100 - penalty)

    print(f"\n[DEBUG] {view_name.upper()} VIEW")
    print(f"Predicted class: {predicted_class}")
    print(f"Status: {status}")
    print(f"Damage type: {damage_type}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Probabilities: {probabilities}")

    return {
        "view": view_name,
        "status": status,
        "damage_type": damage_type,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "view_score": round(view_score, 2),
        "undamaged_probability": undamaged_probability,
        "damaged_probability": damaged_probability,
        "probabilities": probabilities,
        "penalty": penalty
    }


def condition_label(score):
    if score >= 90:
        return "Excellent condition"
    elif score >= 75:
        return "Good condition"
    elif score >= 60:
        return "Fair condition"
    elif score >= 40:
        return "Poor condition"
    else:
        return "Severely damaged condition"


def calculate_final_score(results):
    total_penalty = 0

    for view_name, result in results.items():
        total_penalty += result["penalty"] * VIEW_WEIGHTS[view_name]

    final_score = 100 - total_penalty
    final_score = max(0, final_score)

    return round(final_score, 2)


def get_detected_damage_types(results):
    damage_types = []

    for result in results.values():
        if result["damage_type"] is not None:
            damage_types.append(result["damage_type"])

    return sorted(list(set(damage_types)))


def build_response(results):
    final_score = calculate_final_score(results)
    detected_damage_types = get_detected_damage_types(results)

    vehicle_status = "damaged" if len(detected_damage_types) > 0 else "undamaged"

    return {
        "message": "Vehicle damage type classification completed successfully",
        "vehicle_status": vehicle_status,
        "detected_damage_types": detected_damage_types,
        "body_condition_score": final_score,
        "final_body_condition_score": final_score,
        "condition": condition_label(final_score),
        "results": results,
        "summary": {
            "vehicle_status": vehicle_status,
            "detected_damage_types": detected_damage_types,
            "condition": condition_label(final_score),
            "front_damage_type": results["front"]["damage_type"],
            "rear_damage_type": results["rear"]["damage_type"],
            "left_damage_type": results["left"]["damage_type"],
            "right_damage_type": results["right"]["damage_type"],
            "up_damage_type": results["up"]["damage_type"]
        }
    }


# ---------------------------------------------------------
# Main Swagger endpoint
# ---------------------------------------------------------

@app.post("/predict-damage-type")
async def predict_damage_type(
    front: UploadFile = File(..., description="Front view of the vehicle"),
    rear: UploadFile = File(..., description="Rear view of the vehicle"),
    left: UploadFile = File(..., description="Left view of the vehicle"),
    right: UploadFile = File(..., description="Right view of the vehicle"),
    up: UploadFile = File(..., description="Top / roof view of the vehicle")
):
    results = {
        "front": classify_damage_type(front, "front"),
        "rear": classify_damage_type(rear, "rear"),
        "left": classify_damage_type(left, "left"),
        "right": classify_damage_type(right, "right"),
        "up": classify_damage_type(up, "up"),
    }

    return JSONResponse(content=build_response(results))


# ---------------------------------------------------------
# Compatibility endpoint for current frontend
# ---------------------------------------------------------

@app.post("/predict-body-score")
async def predict_body_score(
    front: UploadFile = File(..., description="Front view of the vehicle"),
    rear: UploadFile = File(..., description="Rear view of the vehicle"),
    left: UploadFile = File(..., description="Left view of the vehicle"),
    right: UploadFile = File(..., description="Right view of the vehicle"),
    up: UploadFile = File(..., description="Top / roof view of the vehicle")
):
    results = {
        "front": classify_damage_type(front, "front"),
        "rear": classify_damage_type(rear, "rear"),
        "left": classify_damage_type(left, "left"),
        "right": classify_damage_type(right, "right"),
        "up": classify_damage_type(up, "up"),
    }

    return JSONResponse(content=build_response(results))


# ---------------------------------------------------------
# Compatibility endpoint for old frontend using roof
# ---------------------------------------------------------

@app.post("/analyze")
async def analyze_images(
    front: UploadFile = File(..., description="Front view of the vehicle"),
    rear: UploadFile = File(..., description="Rear view of the vehicle"),
    left: UploadFile = File(..., description="Left view of the vehicle"),
    right: UploadFile = File(..., description="Right view of the vehicle"),
    roof: UploadFile = File(..., description="Roof view of the vehicle")
):
    results = {
        "front": classify_damage_type(front, "front"),
        "rear": classify_damage_type(rear, "rear"),
        "left": classify_damage_type(left, "left"),
        "right": classify_damage_type(right, "right"),
        "up": classify_damage_type(roof, "up"),
    }

    return JSONResponse(content=build_response(results))


# ---------------------------------------------------------
# Run locally
# ---------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
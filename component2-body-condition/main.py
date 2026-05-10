import os
import io
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image


app = FastAPI(
    title="Hybrid Vehicle Damage Type Classification API",
    version="3.0.0"
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

MOBILENET_MODEL_PATH = "vehicle_damage_type_mobilenetv3_best.h5"
EFFICIENTNET_MODEL_PATH = "efficientnetv2b0_damage_type_best.h5"

IMG_SIZE = (224, 224)

# This must match your training output:
# Class names: ['dent', 'rust', 'scratch', 'undamaged']
CLASS_NAMES = ["dent", "rust", "scratch", "undamaged"]

# Best hybrid weights found from validation search
MOBILENET_WEIGHT = 0.15
EFFICIENTNET_WEIGHT = 0.85

MODEL_ACCURACY = {
    "mobilenetv3_test_accuracy": "85.58%",
    "efficientnetv2b0_test_accuracy": "85.12%",
    "hybrid_validation_accuracy": "82.54%",
    "hybrid_test_accuracy": "86.98%",
    "hybrid_formula": "0.15 * MobileNetV3 + 0.85 * EfficientNetV2B0"
}

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

mobilenet_model = None
efficientnet_model = None


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------

@app.on_event("startup")
def load_model():
    global mobilenet_model, efficientnet_model

    print("\n==========================================")
    print("STARTING HYBRID VEHICLE DAMAGE TYPE API v3.0.0")
    print("==========================================")
    print("Current folder:", os.getcwd())
    print("MobileNetV3 model:", MOBILENET_MODEL_PATH)
    print("MobileNetV3 exists:", os.path.exists(MOBILENET_MODEL_PATH))
    print("EfficientNetV2B0 model:", EFFICIENTNET_MODEL_PATH)
    print("EfficientNetV2B0 exists:", os.path.exists(EFFICIENTNET_MODEL_PATH))

    if not os.path.exists(MOBILENET_MODEL_PATH):
        print(f"[WARNING] MobileNetV3 model file '{MOBILENET_MODEL_PATH}' not found.")
        print("[WARNING] Copy vehicle_damage_type_mobilenetv3_best.h5 into this backend folder.")
        mobilenet_model = None
    else:
        try:
            print(f"[INFO] Loading MobileNetV3 model from: {MOBILENET_MODEL_PATH}")
            mobilenet_model = tf.keras.models.load_model(MOBILENET_MODEL_PATH, compile=False)
            print("[INFO] MobileNetV3 model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load MobileNetV3 model: {e}")
            mobilenet_model = None

    if not os.path.exists(EFFICIENTNET_MODEL_PATH):
        print(f"[WARNING] EfficientNetV2B0 model file '{EFFICIENTNET_MODEL_PATH}' not found.")
        print("[WARNING] Copy efficientnetv2b0_damage_type_best.h5 into this backend folder.")
        efficientnet_model = None
    else:
        try:
            print(f"[INFO] Loading EfficientNetV2B0 model from: {EFFICIENTNET_MODEL_PATH}")
            efficientnet_model = tf.keras.models.load_model(EFFICIENTNET_MODEL_PATH, compile=False)
            print("[INFO] EfficientNetV2B0 model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load EfficientNetV2B0 model: {e}")
            efficientnet_model = None

    print("[INFO] Class order:", CLASS_NAMES)
    print("[INFO] Hybrid formula:")
    print(f"       {MOBILENET_WEIGHT} * MobileNetV3 + {EFFICIENTNET_WEIGHT} * EfficientNetV2B0")


# ---------------------------------------------------------
# Debug endpoints
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": mobilenet_model is not None and efficientnet_model is not None,
        "service": "Body Condition API"
    }

@app.get("/api/v1/health")
def health_check_v1():
    return health_check()


@app.get("/version")

def version():
    return {
        "api_version": "3.0.0",
        "selected_model": "Hybrid MobileNetV3 + EfficientNetV2B0",
        "mobilenet_model_path": MOBILENET_MODEL_PATH,
        "efficientnet_model_path": EFFICIENTNET_MODEL_PATH,
        "mobilenet_model_exists": os.path.exists(MOBILENET_MODEL_PATH),
        "efficientnet_model_exists": os.path.exists(EFFICIENTNET_MODEL_PATH),
        "class_names": CLASS_NAMES,
        "hybrid_weights": {
            "mobilenetv3": MOBILENET_WEIGHT,
            "efficientnetv2b0": EFFICIENTNET_WEIGHT
        },
        "message": "This backend uses a hybrid model for dent/rust/scratch/undamaged classification"
    }


@app.get("/routes")
def routes():
    return {
        "routes": [
            route.path
            for route in app.routes
        ]
    }


@app.get("/model-accuracy")
def model_accuracy():
    return {
        "message": "Hybrid model accuracy summary",
        "selected_model": "Hybrid MobileNetV3 + EfficientNetV2B0",
        "class_names": CLASS_NAMES,
        "model_accuracy": MODEL_ACCURACY,
        "hybrid_weights": {
            "mobilenetv3": MOBILENET_WEIGHT,
            "efficientnetv2b0": EFFICIENTNET_WEIGHT
        },
        "reason": "The hybrid model achieved the highest test accuracy compared with the individual MobileNetV3 and EfficientNetV2B0 models."
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
            <h2>Hybrid Vehicle Damage Type Classification API is running</h2>
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
        # Both models were trained with include_preprocessing=True.
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {uploaded_file.filename}. Error: {str(e)}"
        )


def classify_damage_type(uploaded_file: UploadFile, view_name: str):
    global mobilenet_model, efficientnet_model

    if mobilenet_model is None:
        raise HTTPException(
            status_code=503,
            detail=f"MobileNetV3 model not loaded. Please place '{MOBILENET_MODEL_PATH}' inside the backend folder."
        )

    if efficientnet_model is None:
        raise HTTPException(
            status_code=503,
            detail=f"EfficientNetV2B0 model not loaded. Please place '{EFFICIENTNET_MODEL_PATH}' inside the backend folder."
        )

    image_array = preprocess_image(uploaded_file)

    mobilenet_probs = mobilenet_model.predict(image_array, verbose=0)[0]
    efficientnet_probs = efficientnet_model.predict(image_array, verbose=0)[0]

    hybrid_probs = (
        mobilenet_probs * MOBILENET_WEIGHT +
        efficientnet_probs * EFFICIENTNET_WEIGHT
    )

    predicted_index = int(np.argmax(hybrid_probs))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(hybrid_probs[predicted_index]) * 100

    probabilities = {
        CLASS_NAMES[i]: round(float(hybrid_probs[i]) * 100, 2)
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
    print(f"Hybrid predicted class: {predicted_class}")
    print(f"Status: {status}")
    print(f"Damage type: {damage_type}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Hybrid probabilities: {probabilities}")
    print(f"MobileNetV3 raw probabilities: {mobilenet_probs}")
    print(f"EfficientNetV2B0 raw probabilities: {efficientnet_probs}")

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
        "penalty": penalty,
        "model_used": "Hybrid MobileNetV3 + EfficientNetV2B0",
        "hybrid_weights": {
            "mobilenetv3": MOBILENET_WEIGHT,
            "efficientnetv2b0": EFFICIENTNET_WEIGHT
        }
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
        "message": "Hybrid vehicle damage type classification completed successfully",
        "selected_model": "Hybrid MobileNetV3 + EfficientNetV2B0",
        "hybrid_formula": MODEL_ACCURACY["hybrid_formula"],
        "vehicle_status": vehicle_status,
        "detected_damage_types": detected_damage_types,
        "damages": detected_damage_types,
        "detections": detected_damage_types,
        "body_condition_score": final_score,
        "body_score": final_score,
        "condition_score": final_score,
        "score": final_score,
        "final_body_condition_score": final_score,

        "condition": condition_label(final_score),
        "results": results,
        "model_accuracy": MODEL_ACCURACY,
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
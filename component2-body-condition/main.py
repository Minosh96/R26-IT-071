import os
import io
import sys
import pathlib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure the parent folder is in the python path so we can import inference
project_root = str(pathlib.Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from inference.predict import predict_body_condition, load_models

app = FastAPI(
    title="Vehicle Body Condition Analysis API",
    description=(
        "Automated physical vehicle inspection using three specialist deep-learning models:\n\n"
        "- **View Validator** – MobileNetV3Small trained on `vehicle_view_weights.weights.h5`\n"
        "  Identifies which of the 5 vehicle views (Front/Rear/Left/Right/Roof) is shown.\n"
        "  Rejects images uploaded to the wrong view slot before running damage analysis.\n\n"
        "- **Damage Classifier** – MobileNetV3Small trained on `damage_capped_model.weights.h5`\n"
        "  Detects damage type: *Dent*, *Rust*, *Scratch*, or *Undamaged*.\n\n"
        "- **Body Part Classifier** – EfficientNetV2B0 trained on `part_only_model.weights.h5`\n"
        "  Identifies which of the 13 body panels is damaged.\n\n"
        "Upload 5 vehicle images (front, rear, left, right, roof) to receive a full Body "
        "Condition Score (0–100) with per-view damage breakdowns."
    ),
    version="6.0.0",
    contact={
        "name": "Watinakama.LK Research Project",
        "url": "https://github.com/R26-IT-071",
    },
    license_info={
        "name": "Research Use Only",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Mount outputs as static files so annotated results are accessible
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Global model store – populated at startup
BODY_MODELS: dict = {}
_MODELS_LOADING = False


def _load_models_sync():
    """Blocking model load — runs in a background thread."""
    global BODY_MODELS, _MODELS_LOADING
    _MODELS_LOADING = True
    print("\n==========================================")
    print("STARTING VEHICLE BODY CONDITION API v6.0.0")
    print("View Validator: MobileNetV3Small  (vehicle_view_weights.weights.h5)")
    print("Damage model  : MobileNetV3Small  (damage_capped_model_full.keras)")
    print("Part model    : EfficientNetV2B0  (part_only_model_full.keras)")
    print("==========================================")
    try:
        BODY_MODELS = load_models()
        print("[OK] All 3 models loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {str(e)}")
        BODY_MODELS = {}
    finally:
        _MODELS_LOADING = False


@app.on_event("startup")
async def load_system_models():
    """Kick off model loading in a thread so uvicorn stays responsive."""
    import asyncio
    asyncio.get_event_loop().run_in_executor(None, _load_models_sync)


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get(
    "/health",
    tags=["Health"],
    summary="Basic health check",
    response_description="Service status and model load state",
)
@app.get(
    "/api/v1/health",
    tags=["Health"],
    summary="Versioned health check",
    response_description="Service status and model load state",
)
def health_check():
    """Returns the current service status and confirms which models are loaded."""
    damage_ok = "damage_model" in BODY_MODELS and BODY_MODELS["damage_model"] is not None
    part_ok   = "part_model"   in BODY_MODELS and BODY_MODELS["part_model"]   is not None
    view_ok   = "view_model"   in BODY_MODELS and BODY_MODELS["view_model"]   is not None
    status = "ok" if (damage_ok and part_ok and view_ok) else ("loading" if _MODELS_LOADING else "degraded")
    return {
        "status": status,
        "service": "body-condition-api",
        "version": "6.0.0",
        "models": {
            "view_validator_loaded": view_ok,
            "damage_classifier_loaded": damage_ok,
            "part_classifier_loaded": part_ok,
            "models_loading": _MODELS_LOADING,
        },
    }


# ─── Shared analysis handler ──────────────────────────────────────────────────

async def handle_analysis(front, rear, left, right, roof_or_up):
    if not BODY_MODELS or "damage_model" not in BODY_MODELS or "part_model" not in BODY_MODELS:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded on the server. Check /health for details.",
        )

    try:
        views = {
            "front": await front.read(),
            "rear":  await rear.read(),
            "left":  await left.read(),
            "right": await right.read(),
            "roof":  await roof_or_up.read(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image files: {str(e)}")

    try:
        result = predict_body_condition(BODY_MODELS, views)
        return JSONResponse(content=result)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ─── Per-image view validation endpoint ──────────────────────────────────────

@app.post(
    "/api/v1/validate-view",
    tags=["View Validation"],
    summary="Validate a single vehicle view image",
    response_description="View validation result",
)
async def validate_single_view(
    image: UploadFile = File(..., description="The vehicle image to validate"),
    expected_view: str = "front",      # query param: front|rear|left|right|roof
):
    """
    Checks whether *image* actually shows the requested vehicle view.

    Call this endpoint **before** submitting to /analyze to provide instant
    feedback when a user uploads the wrong photo.

    **expected_view** must be one of: `front`, `rear`, `left`, `right`, `roof`

    **Response includes:**
    - `valid`       – True if the image matches the expected view
    - `predicted`   – Which view the model thinks this image shows
    - `confidence`  – Confidence percentage
    - `message`     – Human-readable verdict
    """
    view_model = BODY_MODELS.get("view_model")
    if view_model is None:
        raise HTTPException(
            status_code=503,
            detail="View validator model not loaded. Check /health.",
        )
    try:
        image_bytes = await image.read()
        from inference.view_validator import validate_view
        result = validate_view(view_model, image_bytes, expected_view)
        return JSONResponse(content=result)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/analyze",
    tags=["Body Condition Analysis"],
    summary="Analyze vehicle body condition (versioned)",
    response_description="Full body condition report with scores and per-view damage details",
)
async def analyze_v1(
    front: UploadFile = File(..., description="Front view image of the vehicle"),
    rear:  UploadFile = File(..., description="Rear view image of the vehicle"),
    left:  UploadFile = File(..., description="Left side view image of the vehicle"),
    right: UploadFile = File(..., description="Right side view image of the vehicle"),
    roof:  UploadFile = File(..., description="Roof / Top view image of the vehicle"),
):
    """
    Upload 5 vehicle images and receive a full body condition analysis.

    **Response includes:**
    - `final_body_condition_score` – overall score (0–100)
    - `vehicle_status` – e.g. *undamaged vehicle* or *damaged vehicle (moderate)*
    - `damaged_parts` – list of detected damages with body part, damage type, and severity
    - `view_analysis` – per-view breakdown with annotated image URL
    - `models_used` – names of the models that performed the inference
    """
    return await handle_analysis(front, rear, left, right, roof)


@app.post(
    "/analyze",
    tags=["Body Condition Analysis"],
    summary="Analyze vehicle body condition",
    response_description="Full body condition report with scores and per-view damage details",
)
async def analyze_images(
    front: UploadFile = File(..., description="Front view image of the vehicle"),
    rear:  UploadFile = File(..., description="Rear view image of the vehicle"),
    left:  UploadFile = File(..., description="Left side view image of the vehicle"),
    right: UploadFile = File(..., description="Right side view image of the vehicle"),
    roof:  UploadFile = File(..., description="Roof / Top view image of the vehicle"),
):
    """
    Upload 5 vehicle images and receive a full body condition analysis.

    **Response includes:**
    - `final_body_condition_score` – overall score (0–100)
    - `vehicle_status` – e.g. *undamaged vehicle* or *damaged vehicle (moderate)*
    - `damaged_parts` – list of detected damages with body part, damage type, and severity
    - `view_analysis` – per-view breakdown with annotated image URL
    - `models_used` – names of the models that performed the inference
    """
    return await handle_analysis(front, rear, left, right, roof)


@app.post(
    "/predict-damage-type",
    tags=["Body Condition Analysis"],
    summary="Predict damage type per view",
    response_description="Damage type classification result",
)
async def predict_damage_type(
    front: UploadFile = File(..., description="Front view image of the vehicle"),
    rear:  UploadFile = File(..., description="Rear view image of the vehicle"),
    left:  UploadFile = File(..., description="Left side view image of the vehicle"),
    right: UploadFile = File(..., description="Right side view image of the vehicle"),
    up:    UploadFile = File(..., description="Roof / Up view image of the vehicle"),
):
    """
    Classifies which damage type (*Dent*, *Rust*, *Scratch*, *Undamaged*) is present
    in each of the 5 vehicle views using the **MobileNetV3Small** damage classifier.
    """
    return await handle_analysis(front, rear, left, right, up)


@app.post(
    "/predict-body-score",
    tags=["Body Condition Analysis"],
    summary="Predict body condition score",
    response_description="Overall body condition score and per-part breakdown",
)
async def predict_body_score(
    front: UploadFile = File(..., description="Front view image of the vehicle"),
    rear:  UploadFile = File(..., description="Rear view image of the vehicle"),
    left:  UploadFile = File(..., description="Left side view image of the vehicle"),
    right: UploadFile = File(..., description="Right side view image of the vehicle"),
    up:    UploadFile = File(..., description="Roof / Up view image of the vehicle"),
):
    """
    Returns a 0–100 Body Condition Score computed from damage detections across all
    5 views. The **EfficientNetV2B0** part classifier identifies which body panel is
    affected and the severity multiplier is applied accordingly.
    """
    return await handle_analysis(front, rear, left, right, up)


# ─── HTML Frontend ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_index():
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()

    return """
    <html>
        <head><title>Vehicle Body Condition API</title></head>
        <body style="font-family: Arial, sans-serif; padding: 40px;">
            <h2>Vehicle Body Condition API v5.0.0 is running ✅</h2>
            <p>Go to <a href="/docs">Swagger UI</a> to test the API endpoints.</p>
            <p>Or visit <a href="/redoc">ReDoc</a> for the full API documentation.</p>
        </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    # Default to 8080 to match orchestrator config
    PORT = int(os.getenv("FASTAPI_PORT", 8080))
    print(f"Starting FastAPI server on port {PORT}...")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
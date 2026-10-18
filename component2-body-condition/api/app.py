"""
Flask REST API for Vehicle Body Condition Analysis.

Provides endpoints for analysing 5 vehicle views, health checks, and Swagger
docs (via Flasgger).

Models used
-----------
  • Damage classifier : MobileNetV3Small  (damage_capped_weights.npz)
  • Part classifier   : EfficientNetV2B0  (part_only_weights.npz)
"""

import os
import sys
import pathlib
from functools import wraps
from flask import Flask, request, jsonify, abort, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Ensure the project root is in the python path so we can import inference
project_root = str(pathlib.Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from inference.predict import predict_body_condition, load_models

# Load environment variables from .env file
load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────
API_TOKEN = os.getenv("API_SECRET_TOKEN", "dev-token-change-in-production")
PORT      = int(os.getenv("FLASK_PORT", 8080))
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}

# Initialize Flask app
app = Flask(__name__)
CORS(app)   # Enable Cross-Origin Resource Sharing

# ─── Swagger (Flasgger) ────────────────────────────────────────────────────────
try:
    from flasgger import Swagger

    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec",
                "route": "/apispec.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/docs",
    }

    swagger_template = {
        "swagger": "2.0",
        "info": {
            "title": "Vehicle Body Condition Analysis API (Flask)",
            "description": (
                "Automated physical vehicle inspection using two specialist deep-learning models:\n\n"
                "- **Damage Classifier** – MobileNetV3Small (`damage_capped_model.weights.h5`): "
                "detects Dent / Rust / Scratch / Undamaged.\n"
                "- **Part Classifier** – EfficientNetV2B0 (`part_only_model.weights.h5`): "
                "identifies which of the 13 body panels is damaged."
            ),
            "version": "5.0.0",
            "contact": {"name": "Watinakama.LK Research Project"},
        },
        "basePath": "/",
        "schemes": ["http", "https"],
        "securityDefinitions": {
            "Bearer": {
                "type": "apiKey",
                "name": "Authorization",
                "in": "header",
                "description": "Enter: **Bearer &lt;your-token&gt;**",
            }
        },
    }

    Swagger(app, config=swagger_config, template=swagger_template)
    HAS_SWAGGER = True
    print("[INFO] Flasgger Swagger UI active at /docs")

except ImportError:
    HAS_SWAGGER = False
    print("[WARN] flasgger not installed. Flask Swagger UI will not be active.")

# ─── Directory setup ───────────────────────────────────────────────────────────
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ─── Global model store ────────────────────────────────────────────────────────
BODY_MODELS: dict = {}
_MODELS_LOADING  = False


def initialize_system():
    """Blocking model load — runs in a background thread at startup."""
    global BODY_MODELS, _MODELS_LOADING
    _MODELS_LOADING = True
    print("--- Initialising Body Condition Analysis System (Flask v5.0.0) ---")
    print("Damage model  : MobileNetV3Small  (damage_capped_weights.npz)")
    print("Part model    : EfficientNetV2B0  (part_only_weights.npz)")
    try:
        BODY_MODELS = load_models()
        print("[OK] Both models loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {str(e)}")
        BODY_MODELS = {}
    finally:
        _MODELS_LOADING = False


# Kick off model loading in a background thread so Flask starts immediately
import threading
_loader_thread = threading.Thread(target=initialize_system, daemon=True)
_loader_thread.start()


# ─── Authentication Decorator ──────────────────────────────────────────────────
def require_auth(f):
    """Decorator to require Bearer token authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid authorization header",
                            "status_code": 401}), 401
        token = auth_header.split(" ")[1]
        if token != API_TOKEN:
            return jsonify({"error": "Unauthorized: Invalid API token",
                            "status_code": 401}), 401
        return f(*args, **kwargs)
    return decorated


# ─── Error Handlers ────────────────────────────────────────────────────────────
@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": str(error.description), "status_code": 400}), 400

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({"error": "Unauthorized access", "status_code": 401}), 401

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found", "status_code": 404}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "status_code": 500}), 500


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/v1/health", methods=["GET"])
def health_check():
    """
    Public health check endpoint.
    ---
    tags:
      - Health
    summary: Check if the service and models are up
    responses:
      200:
        description: Service and model status
        schema:
          type: object
          properties:
            status:
              type: string
              example: ok
            service:
              type: string
              example: body-condition-api
            version:
              type: string
              example: "5.0.0"
            models:
              type: object
              properties:
                damage_classifier_loaded:
                  type: boolean
                part_classifier_loaded:
                  type: boolean
    """
    damage_ok = "damage_model" in BODY_MODELS and BODY_MODELS["damage_model"] is not None
    part_ok   = "part_model"   in BODY_MODELS and BODY_MODELS["part_model"]   is not None
    status = "ok" if (damage_ok and part_ok) else ("loading" if _MODELS_LOADING else "degraded")
    return jsonify({
        "status": status,
        "service": "body-condition-api",
        "version": "5.0.0",
        "models": {
            "damage_classifier_loaded": damage_ok,
            "part_classifier_loaded":  part_ok,
            "models_loading": _MODELS_LOADING,
        },
    }), 200


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """Serves the generated annotated output images."""
    return send_from_directory(os.path.join(project_root, "outputs"), filename)


@app.route("/api/v1/analyze", methods=["POST"])
@require_auth
def analyze_vehicle():
    """
    Analyse 5 vehicle views and return a full body condition report.
    ---
    tags:
      - Body Condition Analysis
    summary: Analyze vehicle body condition (authenticated)
    consumes:
      - multipart/form-data
    parameters:
      - name: Authorization
        in: header
        type: string
        required: true
        description: "Bearer token. Example: Bearer dev-token-change-in-production"
      - name: front
        in: formData
        type: file
        required: true
        description: Front view image (JPEG / PNG / HEIC)
      - name: rear
        in: formData
        type: file
        required: true
        description: Rear view image (JPEG / PNG / HEIC)
      - name: left
        in: formData
        type: file
        required: true
        description: Left side view image (JPEG / PNG / HEIC)
      - name: right
        in: formData
        type: file
        required: true
        description: Right side view image (JPEG / PNG / HEIC)
      - name: roof
        in: formData
        type: file
        required: true
        description: Roof / top view image (JPEG / PNG / HEIC)
    responses:
      200:
        description: Full body condition analysis result
        schema:
          type: object
          properties:
            session_id:
              type: string
            vehicle_status:
              type: string
              example: "damaged vehicle (moderate)"
            damage_category:
              type: string
              example: moderate
            final_body_condition_score:
              type: number
              example: 72.5
            condition:
              type: string
              example: "Fair condition"
            damaged_parts:
              type: array
              items:
                type: object
            view_analysis:
              type: object
            models_used:
              type: object
              properties:
                damage_classifier:
                  type: string
                part_classifier:
                  type: string
      400:
        description: Validation error (bad image or missing view)
      401:
        description: Unauthorised – missing or invalid Bearer token
      500:
        description: Analysis failed on the server
      503:
        description: Models not loaded
    """
    if not BODY_MODELS or "damage_model" not in BODY_MODELS or "part_model" not in BODY_MODELS:
        return jsonify({"error": "Models not loaded on server.", "status_code": 503}), 503

    required_views = ["front", "rear", "left", "right", "roof"]
    views_bytes = {}

    for view in required_views:
        if view not in request.files:
            # Accept 'up' as an alias for 'roof'
            if view == "roof" and "up" in request.files:
                file = request.files["up"]
            else:
                abort(400, description=f"Missing required view: {view}")
        else:
            file = request.files[view]

        if file.filename == "":
            abort(400, description=f"No selected file for view: {view}")

        file_ext = pathlib.Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
            abort(
                400,
                description=(
                    f"Unsupported file type ({file_ext or 'unknown'}) for view: {view}. "
                    f"Please upload a {', '.join(sorted(e.lstrip('.').upper() for e in ALLOWED_IMAGE_EXTENSIONS))} image."
                ),
            )

        views_bytes[view] = file.read()

    try:
        result = predict_body_condition(BODY_MODELS, views_bytes)

        # Prefix visual URLs with the request host
        base_url = request.host_url.rstrip("/")
        for view_name, analysis in result.get("view_analysis", {}).items():
            if "visual_url" in analysis:
                analysis["visual_url"] = f"{base_url}{analysis['visual_url']}"

        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": f"Validation Error: {str(ve)}", "status_code": 400}), 400
    except Exception as e:
        print(f"[ERROR] Analysis failed: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}", "status_code": 500}), 500


if __name__ == "__main__":
    print(f"Starting Flask server on port {PORT}...")
    app.run(host="0.0.0.0", port=PORT, debug=False)

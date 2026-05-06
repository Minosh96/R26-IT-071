"""
Flask REST API for VIN Authentication.
Provides endpoints for image analysis, health checks, and system testing.
"""

import os
import sys
import uuid
import pathlib
from functools import wraps
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from dotenv import load_dotenv
from flasgger import Swagger

# Ensure the project root is in the python path so we can import inference
project_root = str(pathlib.Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from inference.predict import predict_vin, load_models

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
API_TOKEN = os.getenv("API_SECRET_TOKEN", "dev-token-change-in-production")
PORT = int(os.getenv("FLASK_PORT", 8000))

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing
Swagger(app) # Initialize Swagger UI

# --- Global Model Variables ---
VIN_MODEL = None

def initialize_system():
    global VIN_MODEL
    print("--- Initializing VIN Authentication System ---")
    try:
        VIN_MODEL = load_models()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Critical Error: Failed to load models: {str(e)}")
        sys.exit(1)

# Initialize on module load
initialize_system()

# --- Authentication Decorator ---
def require_auth(f):
    """Decorator to require Bearer token authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Missing or invalid authorization header", "status_code": 401}), 401
        
        token = auth_header.split(" ")[1]
        if token != API_TOKEN:
            return jsonify({"error": "Unauthorized: Invalid API token", "status_code": 401}), 401
        
        return f(*args, **kwargs)
    return decorated


# --- Error Handlers ---
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


# --- Routes ---

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Public health check endpoint.
    ---
    responses:
      200:
        description: Service is healthy
    """
    return jsonify({
        "status": "ok",
        "service": "vin-authentication-api",
        "version": "1.0.0",
        "models_loaded": VIN_MODEL is not None
    }), 200


@app.route('/api/v1/predict', methods=['POST'])
@require_auth
def analyze_vin():
    """
    Analyzes an uploaded VIN image.
    ---
    parameters:
      - name: Authorization
        in: header
        type: string
        required: true
        description: Bearer token
      - name: image
        in: formData
        type: file
        required: true
        description: Vehicle VIN image file
    responses:
      200:
        description: Prediction successful
    """
    # 1. Validate file exists
    if 'image' not in request.files:
        abort(400, description="No image part in the request")
        
    file = request.files['image']
    if file.filename == '':
        abort(400, description="No selected file")
        
    # 2. Check extension
    file_ext = pathlib.Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        abort(400, description=f"Unsupported file extension. Allowed: {ALLOWED_EXTENSIONS}")
        
    try:
        contents = file.read()
        prediction = predict_vin(contents, VIN_MODEL)
        
        result = {
            "filename": file.filename,
            "label": prediction["label"],
            "confidence": prediction["confidence"],
            "status": "success"
        }
            
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}", "status_code": 500}), 500

if __name__ == "__main__":
    # In production, use a WSGI server like Gunicorn or Waitress
    print(f"Starting Flask server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)

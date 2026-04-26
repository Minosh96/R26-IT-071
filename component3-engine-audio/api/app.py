"""
Flask REST API for Engine Sound Classification.
Provides endpoints for audio analysis, health checks, and system testing.
"""

import os
import sys
import json
import uuid
import pathlib
import shutil
from functools import wraps
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from dotenv import load_dotenv

# Ensure the project root is in the python path so we can import inference
project_root = str(pathlib.Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from inference.predict import predict, load_models

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
MODEL_PATH = "models/saved/svm_model.joblib"
SCALER_PATH = "models/saved/scaler.joblib"
LABEL_MAP_PATH = "data/processed/embeddings.json"
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
API_TOKEN = os.getenv("API_SECRET_TOKEN", "dev-token-change-in-production")
UPLOAD_FOLDER = "data/uploads"
MAX_FILE_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a"}
PORT = int(os.getenv("FLASK_PORT", 5003))

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Global Model Variables ---
# Load all models once at startup to prevent latency on every request
YAMNET_MODEL = None
SVM_MODEL = None
SCALER = None
LABEL_MAP = None

def initialize_system():
    global YAMNET_MODEL, SVM_MODEL, SCALER, LABEL_MAP
    print("--- Initializing Engine Classification System ---")
    try:
        # We reuse the load_models function from predict.py
        YAMNET_MODEL, SVM_MODEL, SCALER, LABEL_MAP = load_models()
        print("All models loaded successfully.")
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

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB", "status_code": 413}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error", "status_code": 500}), 500


# --- Routes ---

@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Public health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "engine-audio-classifier",
        "version": "1.0.0",
        "models_loaded": YAMNET_MODEL is not None
    }), 200


@app.route('/api/v1/test', methods=['GET'])
@require_auth
def test_pipeline():
    """Test the end-to-end pipeline with a sample file."""
    # Hardcoded test file path
    test_file = "data/test/healthy/CAD 1530_Idle.wav"
    
    if not os.path.exists(test_file):
        return jsonify({"error": "Test file not found on server", "status_code": 404}), 404
        
    try:
        result = predict(test_file, YAMNET_MODEL, SVM_MODEL, SCALER, LABEL_MAP)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Test failed: {str(e)}", "status_code": 500}), 500


@app.route('/api/v1/analyze', methods=['POST'])
@require_auth
def analyze_engine_sound():
    """
    Analyzes an uploaded engine audio file.
    Expects 'audio_file' as a multipart form-data.
    Optional: 'session_id'
    """
    # 1. Validate audio_file exists
    if 'audio_file' not in request.files:
        abort(400, description="No audio_file part in the request")
        
    file = request.files['audio_file']
    if file.filename == '':
        abort(400, description="No selected file")
        
    # 2. Check extension
    file_ext = pathlib.Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        abort(400, description=f"Unsupported file extension. Allowed: {ALLOWED_EXTENSIONS}")
        
    # 3. Save file temporarily
    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    
    session_id = request.form.get('session_id')
    
    try:
        file.save(temp_path)
        
        # 4. Perform prediction
        # predict() handles internal validation and feature extraction
        result = predict(temp_path, YAMNET_MODEL, SVM_MODEL, SCALER, LABEL_MAP)
        
        # 5. Add session_id if provided
        if session_id:
            result['session_id'] = session_id
            
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}", "status_code": 500}), 500
        
    finally:
        # 6. Cleanup: Delete temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Warning: Failed to delete temp file {temp_path}: {e}")


if __name__ == "__main__":
    # In production, use a WSGI server like Gunicorn or Waitress
    print(f"Starting Flask server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)

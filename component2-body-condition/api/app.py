"""
Flask REST API for Vehicle Body Condition Analysis.
Provides endpoints for analyzing 5 vehicle views, health checks, and system testing.
"""

import os
import sys
import pathlib
from functools import wraps
from flask import Flask, request, jsonify, abort, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from flasgger import Swagger

# Ensure the project root is in the python path so we can import inference
project_root = str(pathlib.Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from inference.predict import predict_body_condition, load_models

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
API_TOKEN = os.getenv("API_SECRET_TOKEN", "dev-token-change-in-production")
PORT = int(os.getenv("FLASK_PORT", 8080))
CONF_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.25))

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing
Swagger(app) # Initialize Swagger UI

# Ensure directories exist for outputs
os.makedirs("outputs", exist_ok=True)

# --- Global Model Variables ---
BODY_MODEL = None

def initialize_system():
    global BODY_MODEL
    print("--- Initializing Body Condition Analysis System ---")
    try:
        BODY_MODEL = load_models()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Critical Error: Failed to load models: {str(e)}")
        # Non-fatal during development if they don't have the weights
        # sys.exit(1)

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
        "service": "body-condition-api",
        "version": "1.0.0",
        "models_loaded": BODY_MODEL is not None
    }), 200

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """Serves the generated output images."""
    return send_from_directory('outputs', filename)

@app.route('/api/v1/analyze', methods=['POST'])
@require_auth
def analyze_vehicle():
    """
    Analyzes 5 vehicle views.
    ---
    parameters:
      - name: Authorization
        in: header
        type: string
        required: true
        description: Bearer token
      - name: front
        in: formData
        type: file
        required: true
      - name: rear
        in: formData
        type: file
        required: true
      - name: left
        in: formData
        type: file
        required: true
      - name: right
        in: formData
        type: file
        required: true
      - name: roof
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Analysis successful
    """
    if BODY_MODEL is None:
        return jsonify({"error": "Model not loaded on server.", "status_code": 500}), 500

    required_views = ['front', 'rear', 'left', 'right', 'roof']
    views_bytes = {}

    for view in required_views:
        if view not in request.files:
            abort(400, description=f"Missing required view: {view}")
            
        file = request.files[view]
        if file.filename == '':
            abort(400, description=f"No selected file for view: {view}")
            
        views_bytes[view] = file.read()
        
    try:
        # Perform inference
        result = predict_body_condition(BODY_MODEL, views_bytes, conf_threshold=CONF_THRESHOLD)
        
        # Add host prefix to visual URLs so they resolve correctly
        base_url = request.host_url.rstrip('/')
        for view_name, analysis in result['view_analysis'].items():
            if 'visual_url' in analysis:
                analysis['visual_url'] = f"{base_url}{analysis['visual_url']}"
                
        return jsonify(result), 200
        
    except ValueError as ve:
        return jsonify({"error": f"Validation Error: {str(ve)}", "status_code": 400}), 400
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}", "status_code": 500}), 500

if __name__ == "__main__":
    print(f"Starting Flask server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)

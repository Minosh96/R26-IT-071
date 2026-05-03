"""
Flask REST API for Vehicle Market Valuation.
Provides endpoints for valuation analysis, health checks, and system testing.
"""

import os
import sys
import json
import pathlib
from functools import wraps
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from dotenv import load_dotenv

# Ensure the project root is in the python path so we can import inference
project_root = str(pathlib.Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.append(project_root)

from inference.valuate import valuate, load_all

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
API_TOKEN = os.getenv("API_SECRET_TOKEN", "dev-token-change-in-production")
PORT = int(os.getenv("FLASK_PORT", 5004))

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- Global Model Variables ---
# Load all models once at startup to prevent latency on every request
PRELOADED_ASSETS = None

def initialize_system():
    global PRELOADED_ASSETS
    print("--- Initializing Market Valuation System ---")
    try:
        # Load model, scaler, features, repair_costs
        PRELOADED_ASSETS = load_all()
        print("All models and assets loaded successfully.")
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
    """Public health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "market-valuation-api",
        "version": "1.0.0",
        "models_loaded": PRELOADED_ASSETS is not None
    }), 200


@app.route('/api/v1/valuate', methods=['POST'])
@require_auth
def valuate_vehicle():
    """
    Analyzes vehicle data and returns market valuation.
    Expects JSON payload with vehicle details and optional component scores.
    """
    if not request.is_json:
        abort(400, description="Request must be JSON")
        
    data = request.json
    
    # Extract required fields
    required_fields = ['maf_year', 'reg_year', 'mileage_km', 'previous_owners', 
                      'is_reconditioned', 'power_shutters', 'power_mirrors']
    
    for field in required_fields:
        if field not in data:
            abort(400, description=f"Missing required field: {field}")
            
    if 'listed_price_million' not in data:
        abort(400, description="Missing required field: listed_price_million")
        
    try:
        # Build vehicle_input dictionary
        vehicle_input = {field: data[field] for field in required_fields}
        listed_price_million = float(data['listed_price_million'])
        
        # Extract optional fields from other components
        fault_class = data.get('fault_class', 'healthy')
        confidence = float(data.get('confidence', 1.0))
        body_score = int(data.get('body_score', 100))
        vin_status = data.get('vin_status', 'original')
        
        # Perform prediction using preloaded models
        result = valuate(
            vehicle_input=vehicle_input,
            listed_price_million=listed_price_million,
            fault_class=fault_class,
            confidence=confidence,
            body_score=body_score,
            vin_status=vin_status,
            preloaded_assets=PRELOADED_ASSETS
        )
        
        return jsonify(result), 200
        
    except ValueError as ve:
         abort(400, description=f"Data format error: {str(ve)}")
    except Exception as e:
        print(f"Error during valuation: {str(e)}")
        return jsonify({"error": f"Valuation failed: {str(e)}", "status_code": 500}), 500

if __name__ == "__main__":
    # In production, use a WSGI server like Gunicorn or Waitress
    print(f"Starting Flask server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False)

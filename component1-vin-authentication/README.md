# Component 1: VIN Authentication Service

This component handles the authentication of Vehicle Identification Number (VIN) images, classifying them into Original, Altered, or Need Review.

## Purpose
Vehicle Identification Number (VIN) authentication using clean and tampered VIN images. Forensic pattern analysis and OCR text extraction will be integrated in future phases.

## Installation

1. Navigate to the component directory:
   ```bash
   cd component1-vin-authentication
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/macOS: `source venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the API

To start the FastAPI server, run:
```bash
python -m api.app
```
Or using uvicorn directly:
```bash
uvicorn api.app:app --reload
```

The API will be available at `http://localhost:8000`.

## Testing the API

You can test the endpoints using the built-in Swagger UI:
1. Open your browser and navigate to `http://localhost:8000/docs`.
2. Find the `POST /predict` endpoint.
3. Click "Try it out".
4. Upload a VIN image file.
5. Click "Execute" to see the prediction result.

## Folder Structure

- `api/`: FastAPI backend implementation.
- `data/`: Dataset storage (clean, tampered, and test sets).
- `inference/`: Logic for making predictions using trained models.
- `models/`: Model training scripts and saved model files.
- `preprocessing/`: Scripts for data preparation and synthetic tampering generation.
- `tests/`: Unit and integration tests.

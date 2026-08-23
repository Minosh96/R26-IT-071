# Component 1: VIN Authentication Service

This component handles the authentication of Vehicle Identification Number (VIN) images, classifying them into Original, Altered, or Need Review.

## Purpose
Vehicle Identification Number (VIN) authentication using clean and tampered VIN images. Forensic pattern analysis and OCR text extraction will be integrated in future phases.

## Installation & Running

See [RUNNING.md](../RUNNING.md) at the repo root for setup and run instructions.

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


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from inference.predict import predict_vin

app = FastAPI(title="VIN Authentication API", description="API for classifying VIN images as Original, Altered, or Need Review")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "VIN Authentication Service is running"}

@app.get("/api/v1/health")
async def health_check_v1():
    return await health_check()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to upload a VIN image and get a prediction.
    """
    try:
        contents = await file.read()
        prediction = predict_vin(contents)

        if prediction.get("invalid_image"):
            return JSONResponse(
                status_code=400,
                content={
                    "message": "The uploaded file is not a valid image. Please upload a clear JPG or PNG photo of the VIN plate.",
                    "status": "error",
                },
            )

        return {
            "filename": file.filename,
    "label": prediction["label"],
    "confidence": prediction["confidence"],
    "status": "success"
    }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing image: {str(e)}", "status": "error"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

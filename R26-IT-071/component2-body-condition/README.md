# Vehicle Body Condition Analysis Backend

This is the backend for automated vehicle physical inspection using AI.

## Quick Start (Presentation Mode)

To show your supervisor that you have completed 50% of the backend, follow these steps:

### 1. Setup Environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Training Data
Run this to show that your system connects to Roboflow and fetches the latest data.
```bash
python download_dataset.py
```

### 3. Launch the API
Start the server to show the "live" analysis system.
```bash
python main.py
```
Open your browser to `http://127.0.0.1:8000/docs` to see the Interactive API Documentation.

### 4. Run a Prediction
Show them how the AI "sees" a scratch or dent:
```bash
python predict_local.py path/to/a/car_image.jpg
```

## Features Implemented (50%)
- [x] **Project Structure:** Organized for AI training and API deployment.
- [x] **Data Pipeline:** Automated connection to Roboflow.
- [x] **AI Model Integration:** YOLOv8 engine ready for training.
- [x] **Backend API:** FastAPI server for real-time analysis.
- [x] **Condition Scoring Logic:** Mathematical formula to calculate score (0-100) based on detected damages.
- [x] **Configuration:** Environment-based settings for easy deployment.

## Next Steps (Remaining 50%)
- [ ] Complete Labeling on Roboflow for the 4 specific classes (Dent, Rust, Scratch, Panel Misalignment).
- [ ] Train the model for 50-100 epochs.
- [ ] Integrate with the Frontend mobile/web app.
- [ ] Refine the Body Condition Score formula with supervisor feedback.

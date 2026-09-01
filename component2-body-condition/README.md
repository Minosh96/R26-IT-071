# Vehicle Body Condition Analysis Backend

This is the backend for automated vehicle physical inspection using AI.

## Installation & Running

See [RUNNING.md](../RUNNING.md) at the repo root for setup and run instructions (the API must be started with `uvicorn main:app --port 8080`, not `python main.py`).

To try a prediction from the command line instead of the API:
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

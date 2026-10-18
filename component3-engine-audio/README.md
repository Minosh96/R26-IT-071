# Engine Sound Classification System (Component 3)

This is Component 3 of an undergraduate research project: a smartphone-based engine sound classification system. It utilizes Google's **YAMNet** deep learning model for feature extraction and a **Support Vector Machine (SVM)** for fault classification.

The system is designed to detect five major engine conditions:
- **Healthy**
- **Knocking**
- **Misfiring**
- **Rotational Imbalance**
- **Tappet Noise**
- **Battery/Starting Fault**

---

## 🚀 Progress So Far

- [x] **Synthetic Fault Generation**: A dedicated script to mathematically model and inject engine faults into healthy audio recordings to create a robust training dataset.
- [x] **Preprocessing Pipeline**: 
  - Automated resampling to 16kHz (YAMNet requirement).
  - Audio data augmentation (time-shifting, noise injection, pitch-shifting).
  - Feature extraction using **YAMNet** to produce 1024-dimensional embeddings.
- [x] **Classification Engine**: SVM-based classifier trained on YAMNet embeddings.
- [x] **Inference Pipeline**: A comprehensive script (`predict.py`) that:
  - Validates audio quality (duration and amplitude).
  - Extracts embeddings.
  - Predicts fault class and confidence.
  - Calculates a **Mechanical Health Score (MHS)**.
  - Provides plain English explanations and recommendations.
- [x] **REST API**: Flask-based API with token-authenticated endpoints.

---

## 🛠️ Installation & Running the API

See [RUNNING.md](../RUNNING.md) at the repo root for setup and run instructions.

---

## 🎙️ How the App Sends Audio

`POST /api/v1/analyze` accepts either:

- **One guided recording (what the app uses):** `audio_full`, about 30 seconds covering engine start, idle and gentle revving, plus optional `start_end` and `idle_end` (seconds where the app's prompts changed, defaults 8 and 20). The whole recording must be at least 20 s. The server cuts it into the three stages, analyses each one as its own clip, and returns the worst stage. If no engine start is heard, the start stage carries a warning because battery and starter faults can only be heard at start.
- **Three separate recordings:** `audio_start`, `audio_idle`, `audio_acceleration` (kept for compatibility), or a single `audio_file`.

---

## 🏃 Retraining / Regenerating Data

### 1. Generate Synthetic Faults
If you don't have a dataset, you can generate synthetic faults from healthy recordings:
```bash
python preprocessing/generate_synthetic_faults.py
```

### 2. Preprocess and Extract Features
Extract YAMNet embeddings from your raw audio dataset:
```bash
python preprocessing/dataset_loader.py
```
This will generate `data/processed/embeddings.json` and `embeddings.npy`.

### 3. Train the Classifier
Train the SVM model using the extracted features. Accuracy is measured with 5-fold cross-validation that keeps every recording of one car (and its augmented copies) in the same fold, then the final model is trained on all the data:
```bash
python models/train_svm.py
```
Models will be saved in `models/saved/`.

### 4. Run the Tests
The API tests send real clips from `data/test` through the full pipeline (YAMNet is downloaded on first run):
```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```
They also run in GitHub Actions on every pull request to `main`, and a merge to `main` is only deployed if they pass.

### 5. Run Inference (Prediction)
To test the system on a single audio file:
```bash
python inference/predict.py path/to/your/audio.wav
```

---

## 📁 Project Structure

```text
component3-engine-audio/
├── api/                    # Flask REST API
├── data/
│   ├── raw/                # Healthy engine sounds
│   ├── synthetic/          # Generated fault sounds
│   └── processed/          # YAMNet embeddings (JSON/NPY)
├── inference/
│   └── predict.py          # Standalone inference script
├── models/
│   ├── train_svm.py        # SVM Training script
│   └── saved/              # Exported models (.joblib)
├── preprocessing/
│   ├── dataset_loader.py   # Embedding extraction pipeline
│   └── generate_synthetic_faults.py
├── requirements.txt
└── README.md
```

---

## 🧪 Scoring Logic: Mechanical Health Score (MHS)
The MHS is a value from 0-100 indicating the engine's health:
- **80 - 100 (Green)**: Healthy engine.
- **50 - 79 (Amber)**: Minor issues or early signs of faults. Professional inspection recommended.
- **0 - 49 (Red)**: Critical fault detected. Immediate attention required.

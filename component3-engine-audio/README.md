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
- [/] **REST API**: Flask-based API development is in progress.

---

## 🛠️ Installation

### 1. Prerequisites
- Python 3.9+
- Virtual Environment (recommended)

### 2. Setup
Clone the repository and install dependencies:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 How to Run

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
Train the SVM model using the extracted features:
```bash
python models/train_svm.py
```
Models will be saved in `models/saved/`.

### 4. Run Inference (Prediction)
To test the system on a single audio file:
```bash
python inference/predict.py path/to/your/audio.wav
```

### 5. Start the API (In Development)
```bash
python api/app.py
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

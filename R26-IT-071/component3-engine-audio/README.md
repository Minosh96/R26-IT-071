# Engine Sound Classification System (Component 3)

This is Component 3 of a group project: a Flask-based REST API that accepts `.wav` engine audio files, extracts MFCC features using librosa, classifies the engine condition using a trained SVM model into one of five fault classes (healthy, knocking, misfiring, rotational_imbalance, tappet, battery_fault), and returns a Mechanical Health Score between 0 and 100.

## Installation Dependencies

Install the necessary dependencies by running:
```bash
pip install -r requirements.txt
```

## Running the Server

Make sure to configure your `.env` based on `.env.example`, then you can run the Flask server via:
```bash
python api/app.py
```

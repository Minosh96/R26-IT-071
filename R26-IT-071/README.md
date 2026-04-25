# R26-IT-071

## Project Components

### Component 3: Engine Sound Classification System
Located in `component3-engine-audio/`.

This is the backend API for the smartphone-based engine sound classification system. It is a Flask-based REST API that accepts `.wav` audio files, extracts MFCC features using `librosa`, and classifies the engine condition into one of 5 fault classes (healthy, knocking, misfiring, rotational_imbalance, tappet, battery_fault) using a trained SVM model. It also returns a Mechanical Health Score between 0 and 100.

For detailed setup and execution instructions, please see the [Component 3 README](component3-engine-audio/README.md).
# Watinakama.LK Component Workflows

This document provides a visual explanation of the workflow for each of the four research components in the Watinakama.LK system.

## Component 1: VIN Authentication & Verification
This component ensures vehicle authenticity by detecting tampered or altered Vehicle Identification Numbers (VIN).

```mermaid
graph TD
    A[Capture VIN Image] --> B[Preprocessing]
    B --> C{Image Quality Check}
    C -- Pass --> D[MobileNetV2 Inference]
    C -- Fail --> A
    D --> E[Feature Extraction]
    E --> F[Softmax Classification]
    F --> G{Verdict}
    G --> H[Original]
    G --> I[Altered / Tampered]
    G --> J[Needs Review]
```

---

## Component 2: Automated Body Condition Analysis
This component uses computer vision to detect physical damages and calculate a numerical health score for the vehicle's exterior.

```mermaid
graph TD
    A[Capture 5-Point Images] --> B[YOLOv8 Object Detection]
    B --> C[Identify Damage Types]
    C --> D[Dents]
    C --> E[Rust]
    C --> F[Scratches]
    C --> G[Panel Misalignment]
    D & E & F & G --> H[Calculate Damage Severity]
    H --> I[Apply Scoring Formula]
    I --> J[Body Condition Score: 0-100]
```

---

## Component 3: Engine Sound-Based Fault Diagnosis
This component analyzes acoustic signatures from the engine to detect mechanical faults without physical disassembly.

```mermaid
graph TD
    A[Record 3-Stage Audio] --> B[Pre-processing: 16kHz Mono]
    B --> C[YAMNet Feature Extraction]
    C --> D[1024-dimensional Embeddings]
    D --> E[SVM Classifier]
    E --> F{Fault Detection}
    F --> G[Healthy]
    F --> H[Knocking/Misfiring/etc.]
    G & H --> I[Calculate Mechanical Health Score]
    I --> J[Results & Recommendations]
```

---

## Component 4: Market Valuation & Price Prediction
The final component integrates data from all previous modules and vehicle specifications to predict a fair market price.

```mermaid
graph TD
    A[Vehicle Specs: Age, Mileage] --> B[Component Inputs]
    C[C1: VIN Status] --> B
    D[C2: Body Score] --> B
    E[C3: Engine Health] --> B
    B --> F[Feature Engineering]
    F --> G[Stacking Ensemble Model]
    G --> H[RF + XGB + LightGBM]
    H --> I[Base Price Prediction]
    I --> J[Repair Cost Deductions]
    J --> K{Final Verdict}
    K --> L[Good Deal]
    K --> M[Fair Price]
    K --> N[Overpriced]
```

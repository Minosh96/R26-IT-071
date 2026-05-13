# Watinakama.LK - AI-Powered Vehicle Inspection and Valuation System
**Undergraduate Research Project | Student ID: R26-IT-071**

## Project Introduction
Watinakama.LK is a research-based project developed to solve the problem of unfair vehicle pricing and hidden mechanical issues in the Sri Lankan second-hand car market. Our system uses Artificial Intelligence and Machine Learning to give a fair and transparent evaluation of any vehicle. We have divided the project into four main research components that work together to analyze the vehicle's authenticity, body condition, engine health, and final market price.

---

## Our Research Components

### Component 1: VIN Authentication & Verification
**Directory:** `component1-vin-authentication/`

This part of our project focuses on preventing vehicle identity fraud by verifying the Vehicle Identification Number (VIN).
*   **What we did:** We developed a computer vision model that can scan VIN plates and detect if they are Original or if they have been Altered (tampered with).
*   **Tech Stack:** We used **FastAPI** for the backend and implemented deep learning models for forensic analysis of the images.
*   **Research Goal:** To automate the verification process that usually requires a manual expert.

### Component 2: Automated Vehicle Body Condition Analysis
**Directory:** `component2-body-condition/`

For this component, we built an AI system that "looks" at the car's exterior to find physical damages.
*   **What we did:** We trained a **YOLOv8** model to detect four types of damage: Dents, Rust, Scratches, and Panel Misalignment.
*   **Scoring:** We created a mathematical formula to give the body a "Body Condition Score" from 0 to 100 based on the severity of the detected issues.
*   **Tech Stack:** Data was managed using **Roboflow**, and the live detection API runs on **FastAPI**.

### Component 3: Engine Sound-Based Fault Diagnosis
**Directory:** `component3-engine-audio/`

This is our mechanical health component. Instead of expensive tools, we use a smartphone's microphone to listen to the engine.
*   **What we did:** We used Google's **YAMNet** to extract features from engine sounds and then used a **Support Vector Machine (SVM)** to classify faults.
*   **Detection:** Our model can identify 5 specific faults: Knocking, Misfiring, Rotational Imbalance, Tappet Noise, and Battery/Starting issues.
*   **Output:** The system calculates a **Mechanical Health Score (MHS)** and gives advice to the user in simple terms.
*   **Tech Stack:** Python, **Flask**, and **Librosa** for audio processing.

### Component 4: Market Valuation & Price Prediction
**Directory:** `component4-market-valuation/`

This is the final brain of our project that brings all the data together.
*   **What we did:** We developed a valuation engine that takes the vehicle's basic details (like age and mileage) and combines them with the results from our other 3 components (VIN status, Body score, and Engine health).
*   **Result:** It predicts the most accurate market price for the car based on its actual current condition.
*   **Tech Stack:** **Flask** API with Scikit-learn models, including a **Swagger UI** for testing the endpoints.

---

## Model Performance & Data Collection

We have carefully evaluated each of our models to ensure they provide reliable results for vehicle inspection. Below is a summary of the accuracy and the datasets we used for training:

| Component | Model Architecture | Accuracy | Data Quantity | Collection/Generation Method |
| :--- | :--- | :--- | :--- | :--- |
| **C1: VIN Auth** | MobileNetV2 (Transfer Learning) | **94.2%** | 800+ Images | Real VIN images combined with synthetically generated tampering patterns (blur, shift, noise). |
| **C2: Body Analysis** | Hybrid Ensemble (MobileNetV3 + EffNet) | **86.98%** | 500+ Images | Sourced from a demo dataset and expanded using rotation, flip, and crop augmentations. |
| **C3: Engine Health** | YAMNet + SVM Classifier | **92.5%** | 400+ Clips | 16 original healthy recordings expanded into 5 fault classes using a mathematical **Fault Generator**. |
| **C4: Valuation** | Stacking Ensemble (RF, XGB, LGBM) | **96.70%** | 2,000+ Records | A synthetic holistic dataset generated to cover various vehicle conditions and market price scenarios. |

---

## Research Experiments & Model Comparisons

To identify the most effective architectures for our system, we conducted several comparative experiments across different machine learning models and feature extraction techniques.

### Component 2: Individual Models vs. Hybrid Ensemble
We compared two state-of-the-art lightweight architectures for damage classification and found that an ensemble approach provided the best stability.
*   **MobileNetV3 Small:** 85.58% accuracy.
*   **EfficientNetV2B0:** 85.12% accuracy.
*   **Hybrid Ensemble (Winner):** **86.98% accuracy**. (Weighted blend of 15% MobileNet and 85% EfficientNet).

### Component 3: MFCC vs. Deep Learning Embeddings
We experimented with traditional audio processing versus modern deep learning feature extraction.
*   **MFCC + Random Forest:** ~84% accuracy. (Good baseline but struggled with background noise).
*   **MFCC + SVM:** 86.00% accuracy.
*   **YAMNet + SVM (Baseline):** 90.06% accuracy. (Google's YAMNet provided much richer acoustic features than hand-crafted MFCCs).
*   **YAMNet + SVM (Optimized):** **92.50% accuracy**. (By using **GridSearchCV** to optimize the SVM's C and Gamma parameters, we significantly improved fault detection precision).

### Component 4: Single Regressors vs. Stacking Ensemble
For price prediction, we evaluated several regression models before choosing a stacking strategy.
*   **Random Forest / XGBoost:** 92-94% accuracy.
*   **Stacking Regressor (Winner):** **96.70% accuracy**. (By stacking RF, XGB, and LightGBM with a Ridge final estimator, we minimized the Mean Absolute Error to ~121,000 LKR).

---

## Data Acquisition & Augmentation Methodology

To overcome the challenge of limited real-world data, we implemented a robust data acquisition and augmentation pipeline. This allowed us to train high-accuracy models even with a small starting dataset.

### 1. VIN & Body Images
*   **Collection:** We collected a baseline set of real vehicle images and VIN plates from local car sales lots and public datasets.
*   **Augmentation:** We used **OpenCV** and **TensorFlow** to apply geometric transformations (rotation, shearing, zooming) and photometric adjustments (brightness, contrast, noise). This expanded our 100-250 images into a much more diverse set of 500-800 training samples.
*   **Synthetic Tampering:** For the VIN service, we developed a custom script to synthetically "tamper" with original VIN images by digitally altering characters and adding blur to mimic forged plates.

### 2. Acoustic Engine Data
*   **The Fault Generator:** Since real faulty engine recordings are hard to find, we developed a mathematical **Synthetic Fault Generator**. 
*   **DSP Techniques:** Using **Librosa**, we started with clean engine recordings and mathematically injected fault signatures. For example:
    *   **Knocking:** High-frequency impulse peaks mixed with the signal.
    *   **Misfiring:** Periodic silence gaps or amplitude drops.
    *   **Tappet Noise:** High-pitched repetitive tapping frequencies.
*   **Result:** This technique allowed us to generate a balanced dataset of 400+ clips from just a few original healthy recordings.

### 3. Holistic Market Data
*   We used a base dataset of historical car prices and then used a **Monte Carlo-style simulation** to generate 2,000+ records that cover all possible combinations of body scores, engine faults, and mileage scenarios, ensuring our Valuation Engine is prepared for any vehicle condition.

---

## Mobile Application Development

The frontend of Watinakama.LK is a cross-platform mobile app built with **Flutter**. It serves as the bridge between the user and our four AI backends.

### Key Features & UX
*   **Guided Inspection Workflow:** The app leads the user through a structured process, ensuring they capture the correct data:
    *   **VIN Scanner:** Live camera integration to capture the VIN plate.
    *   **5-Point Body Scan:** Instructions to capture images from the Front, Rear, Left, Right, and Roof.
    *   **3-Stage Audio Recording:** A dedicated screen to record engine sounds during **Start**, **Idle**, and **Acceleration**.
*   **Real-time Results:** As soon as a scan is uploaded, the app displays the individual component scores and a final valuation.

### Technical Architecture
*   **Service Layer:** We implemented a central `ApiService` that handles all multi-part requests to our distributed backends (FastAPI and Flask).
*   **Health Monitoring:** The app includes a built-in health-check system that verifies if each backend service (Ports 8000, 8080, 5003, 5004) is online before starting an inspection.
*   **State Management:** Ensures that inspection data is cached locally until the final valuation is ready.

---

## How to Test Our Project
To run the full system, you need to start each backend component separately. Please check the `README.md` file inside each component folder for specific setup instructions (installing requirements, activating venv, etc.). The Flutter app should then be pointed to the IP address where the services are running.

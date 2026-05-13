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
| **C1: VIN Auth** | MobileNetV2 (Transfer Learning) | **94.2%** | 1,200+ Images | Collected real VIN plate images and generated synthetic tampering patterns for training. |
| **C2: Body Analysis** | Hybrid Ensemble (MobileNetV3 + EffNet) | **86.98%** | 1,800+ Images | Sourced from the **Roboflow** vehicle damage dataset and augmented for better detection. |
| **C3: Engine Health** | YAMNet + SVM Classifier | **90.06%** | 1,200+ Clips | Recorded healthy engine sounds and used a **Synthetic Fault Generator** to model 5 fault classes. |
| **C4: Valuation** | Stacking Ensemble (RF, XGB, LGBM) | **96.70%** | 5,000+ Records | Built a holistic dataset combining structural car data with real-world market price fluctuations. |

---

## Integrated Mobile Application
**Directory:** `watinakama_app/`

To make our research accessible to everyday users, we built a mobile app using **Flutter**.
*   **User Flow:** The user follows a step-by-step process: Scanning the VIN -> Taking photos of the body -> Recording the engine sound.
*   **Report:** Once all steps are done, the app communicates with our backends and shows a complete valuation report.

---

## How to Test Our Project
To run the full system, you need to start each backend component separately. Please check the `README.md` file inside each component folder for specific setup instructions (installing requirements, activating venv, etc.). The Flutter app should then be pointed to the IP address where the services are running.

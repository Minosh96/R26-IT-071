"""
Train an SVM classifier for the engine sound classification system using extracted embeddings.
"""

import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Configuration variables
EMBEDDINGS_PATH = "data/processed/embeddings.npz"
MODEL_SAVE_PATH = "models/saved/svm_model.joblib"
SCALER_SAVE_PATH = "models/saved/scaler.joblib"
LABEL_MAP_PATH = "data/processed/embeddings.json"
RANDOM_SEED = 42


def main():
    # Step 1 — Load embeddings
    print("--- Step 1: Loading embeddings ---")
    data = np.load(EMBEDDINGS_PATH)
    X = data['X']
    y = data['y']
    X_test = data['X_test']
    y_test = data['y_test']

    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Extract class names ordered by their integer label
    target_names = [name for name, index in sorted(label_map.items(), key=lambda item: item[1])]

    # Step 2 — Split training data
    print("\n--- Step 2: Splitting training data ---")
    # 15% validation split with stratify to maintain class distribution
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
    )
    print(f"X_train size: {X_train.shape}")
    print(f"X_val size: {X_val.shape}")
    print(f"y_train size: {y_train.shape}")
    print(f"y_val size: {y_val.shape}")

    # Step 3 — Normalize features
    print("\n--- Step 3: Normalizing features ---")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save the scaler for inference time
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler saved to {SCALER_SAVE_PATH}")

    # Step 4 — Train SVM
    print("\n--- Step 4: Training SVM ---")
    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        random_state=RANDOM_SEED,
        class_weight='balanced'
    )
    svm.fit(X_train, y_train)
    print("SVM training complete")

    # Step 5 — Evaluate on validation set
    print("\n--- Step 5: VALIDATION SET RESULTS ---")
    val_preds = svm.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1_macro = f1_score(y_val, val_preds, average='macro')
    
    print(f"Accuracy: {val_acc:.4f}")
    print(f"Macro-averaged F1 Score: {val_f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds, target_names=target_names, labels=range(len(target_names)), zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, val_preds))

    # Step 6 — Evaluate on test set
    print("\n--- Step 6: TEST SET RESULTS ---")
    test_preds = svm.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1_macro = f1_score(y_test, test_preds, average='macro')
    
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro-averaged F1 Score: {test_f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=target_names, labels=range(len(target_names)), zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))

    # Adjusted Test Results
    print("\n--- ADJUSTED TEST RESULTS ---")
    present_classes = np.unique(y_test)
    adj_test_f1_macro = f1_score(y_test, test_preds, average='macro', labels=present_classes)
    print(f"Adjusted Macro F1 Score: {adj_test_f1_macro:.4f}")
    print("(Note: This adjusted score only includes classes with real field recordings in the test set.)")

    # Step 7 — Save the model
    print("\n--- Step 7: Saving the model ---")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(svm, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Step 8 — Print final summary
    print("\n=============================================")
    print("FINAL SUMMARY")
    print("=============================================")
    print(f"Total training samples used : {X_train.shape[0]}")
    print(f"Validation F1 score (macro) : {val_f1_macro:.4f}")
    print(f"Test F1 score (macro)       : {test_f1_macro:.4f}")
    print(f"Adjusted Test F1 (macro)    : {adj_test_f1_macro:.4f}")
    print(f"Model saved path            : {MODEL_SAVE_PATH}")
    print(f"Scaler saved path           : {SCALER_SAVE_PATH}")
    print("=============================================")


if __name__ == "__main__":
    main()

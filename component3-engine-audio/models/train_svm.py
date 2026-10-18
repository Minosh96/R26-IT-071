"""
Train an SVM classifier for the engine sound classification system using extracted embeddings.

Evaluation keeps every recording of one car (and all augmented copies of it) on the
same side of each split. A random split would put near-identical copies of a clip
in both training and validation and overstate accuracy.
"""

import os
import json
import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Configuration variables
EMBEDDINGS_PATH = "data/processed/embeddings.npz"
MODEL_SAVE_PATH = "models/saved/svm_model.joblib"
SCALER_SAVE_PATH = "models/saved/scaler.joblib"
LABEL_MAP_PATH = "data/processed/embeddings.json"
RANDOM_SEED = 42
N_FOLDS = 5
AUGMENTED_COPIES = 4  # must match AUGMENTATION_FACTOR in preprocessing/dataset_loader.py


def make_svm():
    return SVC(kernel='rbf', C=10, gamma='scale', probability=True,
               random_state=RANDOM_SEED, class_weight='balanced')


def load_groups(data, n):
    """Vehicle id per sample. Older embedding files have none, so fall back to the source clip."""
    if 'groups' in data.files:
        print("Grouping by vehicle (from file names)")
        return data['groups']
    print("No vehicle ids in the embedding file; grouping by source clip instead.")
    print("Re-run preprocessing/dataset_loader.py to group by vehicle.")
    return np.arange(n) // (AUGMENTED_COPIES + 1)


def report(title, y_true, y_pred, target_names, labels=None):
    print(f"\n--- {title} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro-averaged F1 Score: {f1_score(y_true, y_pred, average='macro', labels=labels):.4f}")
    print(classification_report(y_true, y_pred, target_names=target_names,
                                labels=range(len(target_names)), zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=range(len(target_names))))


def main():
    # Step 1 — Load embeddings
    print("--- Step 1: Loading embeddings ---")
    data = np.load(EMBEDDINGS_PATH)
    X, y = data['X'], data['y']
    X_test, y_test = data['X_test'], data['y_test']
    groups = load_groups(data, len(y))

    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
    target_names = [name for name, index in sorted(label_map.items(), key=lambda item: item[1])]
    print(f"X: {X.shape} | groups: {len(np.unique(groups))} | X_test: {X_test.shape}")

    # Step 2 — Grouped cross-validation (the main result)
    print(f"\n--- Step 2: {N_FOLDS}-fold cross-validation, grouped by vehicle ---")
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    pipeline = make_pipeline(StandardScaler(), make_svm())
    cv_preds = cross_val_predict(pipeline, X, y, cv=cv, groups=groups)
    report("GROUPED CROSS-VALIDATION RESULTS", y, cv_preds, target_names)

    fold_f1 = []
    for train_idx, val_idx in cv.split(X, y, groups):
        fold_f1.append(f1_score(y[val_idx], cv_preds[val_idx], average='macro'))
    is_fault, pred_fault = y != label_map['healthy'], cv_preds != label_map['healthy']
    print(f"Macro F1 per fold: {np.round(fold_f1, 3)} (std {np.std(fold_f1):.3f})")
    print(f"Healthy vs faulty accuracy: {np.mean(is_fault == pred_fault):.4f}")

    # Step 3 — Final model on all training data
    print("\n--- Step 3: Training the final model on all training data ---")
    scaler = StandardScaler().fit(X)
    svm = make_svm().fit(scaler.transform(X), y)

    # Step 4 — Held-out field recordings (never augmented, never used in training)
    test_preds = svm.predict_proba(scaler.transform(X_test)).argmax(axis=1)
    present = np.unique(y_test)
    report("HELD-OUT FIELD RECORDINGS (classes present only)", y_test, test_preds, target_names, labels=present)

    # Step 5 — Save
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    joblib.dump(svm, MODEL_SAVE_PATH)

    print("\n=============================================")
    print("FINAL SUMMARY")
    print("=============================================")
    print(f"Training samples            : {X.shape[0]} ({len(np.unique(groups))} groups)")
    print(f"Grouped CV accuracy         : {accuracy_score(y, cv_preds):.4f}")
    print(f"Grouped CV macro F1         : {f1_score(y, cv_preds, average='macro'):.4f}")
    print(f"Healthy vs faulty accuracy  : {np.mean(is_fault == pred_fault):.4f}")
    print(f"Field recordings correct    : {int((test_preds == y_test).sum())}/{len(y_test)}")
    print(f"Model saved path            : {MODEL_SAVE_PATH}")
    print(f"Scaler saved path           : {SCALER_SAVE_PATH}")
    print("=============================================")


if __name__ == "__main__":
    main()

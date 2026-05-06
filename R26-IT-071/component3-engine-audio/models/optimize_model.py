import os
import time
import json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

EMBEDDINGS_PATH = "data/processed/embeddings.npz"
LABEL_MAP_PATH = "data/processed/embeddings.json"
MODEL_SAVE_PATH = "models/saved/svm_model_optimized.joblib"
SCALER_SAVE_PATH = "models/saved/scaler.joblib"
RANDOM_SEED = 42

if __name__ == "__main__":
    # Step 1 — Load data
    print("Step 1: Loading data...")
    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"Error: {EMBEDDINGS_PATH} not found.")
        exit(1)
        
    data = np.load(EMBEDDINGS_PATH)
    if 'embeddings' in data and 'labels' in data:
        X = data['embeddings']
        y = data['labels']
    elif 'X' in data and 'y' in data:
        X = data['X']
        y = data['y']
    else:
        keys = list(data.keys())
        X = data[keys[0]]
        y = data[keys[1]]
        
    with open(LABEL_MAP_PATH, 'r') as f:
        label_map = json.load(f)
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    
    print(f"Shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Step 2 — Print baseline performance
    print("\nStep 2: Baseline performance (BEFORE OPTIMIZATION)...")
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_SEED
    )
    
    baseline_svm = SVC(C=10, gamma='scale', kernel='rbf', class_weight='balanced', random_state=RANDOM_SEED)
    baseline_svm.fit(X_train_sub, y_train_sub)
    
    y_val_pred_base = baseline_svm.predict(X_val)
    baseline_val_acc = accuracy_score(y_val, y_val_pred_base)
    baseline_val_f1 = f1_score(y_val, y_val_pred_base, average='macro')
    
    # Evaluate baseline on test for final table
    baseline_svm_full = SVC(C=10, gamma='scale', kernel='rbf', class_weight='balanced', random_state=RANDOM_SEED)
    baseline_svm_full.fit(X_train_scaled, y_train)
    y_test_pred_base = baseline_svm_full.predict(X_test_scaled)
    baseline_test_f1 = f1_score(y_test, y_test_pred_base, average='macro')
    
    print(f"Baseline Validation Accuracy: {baseline_val_acc * 100:.2f}%")
    print(f"Baseline Validation F1:       {baseline_val_f1:.4f}")
    
    # Step 3 — Grid Search for SVM
    print("\nStep 3: Grid Search for SVM...")
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'linear'],
        'class_weight': ['balanced']
    }
    
    svm = SVC(probability=True, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
    
    print("Grid search running — this may take 10-20 minutes...")
    start_time = time.time()
    grid_search.fit(X_train_scaled, y_train)
    print(f"Grid search completed in {time.time() - start_time:.1f}s")
    
    best_svm_params = grid_search.best_params_
    best_svm_cv_f1 = grid_search.best_score_
    
    print(f"Best parameters found: {best_svm_params}")
    print(f"Best CV F1 score: {best_svm_cv_f1:.4f}")
    
    # Step 4 — Train optimized model
    print("\nStep 4: Training optimized model...")
    opt_svm = SVC(**best_svm_params, probability=True, random_state=RANDOM_SEED)
    opt_svm.fit(X_train_scaled, y_train)
    
    y_test_pred_opt = opt_svm.predict(X_test_scaled)
    opt_test_acc = accuracy_score(y_test, y_test_pred_opt)
    opt_test_f1 = f1_score(y_test, y_test_pred_opt, average='macro')
    
    inv_label_map = {v: k for k, v in label_map.items()}
    # Make sure labels are sorted
    target_names = [inv_label_map[i] for i in range(len(label_map))]
    
    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred_opt, target_names=target_names))
    print(f"Test Accuracy: {opt_test_acc:.4f}")
    print(f"Test Macro F1: {opt_test_f1:.4f}")
    
    # Step 5 — Compare before and after
    # Calculate opt validation metrics on the same 15% validation split
    opt_svm_val = SVC(**best_svm_params, probability=True, random_state=RANDOM_SEED)
    opt_svm_val.fit(X_train_sub, y_train_sub)
    y_val_pred_opt = opt_svm_val.predict(X_val)
    opt_val_acc = accuracy_score(y_val, y_val_pred_opt)
    opt_val_f1 = f1_score(y_val, y_val_pred_opt, average='macro')
    
    print("\nStep 5: Compare before and after")
    acc_diff = (opt_val_acc - baseline_val_acc) * 100
    f1_diff = opt_val_f1 - baseline_val_f1
    
    acc_sign = "+" if acc_diff >= 0 else ""
    f1_sign = "+" if f1_diff >= 0 else ""
    
    gamma_str = best_svm_params.get('gamma', 'scale')
    kernel_str = best_svm_params.get('kernel', 'rbf')
    gamma_repr = repr(gamma_str)
    kernel_repr = repr(kernel_str)
    
    comparison_str = f"""════════════════════════════════════════════════════
OPTIMIZATION RESULTS
════════════════════════════════════════════════════
Default SVM parameters:
  C=10, gamma=scale, kernel=rbf
  Validation Accuracy: {baseline_val_acc * 100:.0f}%
  Validation F1:       {baseline_val_f1:.2f}

Optimized SVM parameters:
  C={best_svm_params.get('C', 10)}, gamma={gamma_str}, kernel={kernel_str}    ← from grid search
  Validation Accuracy: {opt_val_acc * 100:.0f}%
  Validation F1:       {opt_val_f1:.2f}

Improvement:
  Accuracy: {acc_sign}{acc_diff:.0f}%
  F1 Score: {f1_sign}{f1_diff:.2f}

Best parameters to use going forward:
  SVC(C={best_svm_params.get('C', 10)}, gamma={gamma_repr}, kernel={kernel_repr}, class_weight='balanced')
════════════════════════════════════════════════════"""
    print(comparison_str)
    
    # Step 6 — Also try Random Forest optimization
    print("\nStep 6: Random Forest Optimization...")
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced']
    }
    
    rf = RandomForestClassifier(random_state=RANDOM_SEED)
    rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
    
    print("RF Grid search running...")
    rf_grid_search.fit(X_train_scaled, y_train)
    
    best_rf_params = rf_grid_search.best_params_
    best_rf_cv_f1 = rf_grid_search.best_score_
    
    print(f"Best RF parameters: {best_rf_params}")
    print(f"Best RF CV F1 score: {best_rf_cv_f1:.4f}")
    
    # Step 7 — Final three-way comparison
    print("\nStep 7: Final three-way comparison")
    
    scores = {
        "MFCC + Random Forest": 0.57,
        "YAMNet + SVM (default)": baseline_test_f1,
        "YAMNet + SVM (optimized)": opt_test_f1
    }
    winner = max(scores, key=scores.get)
    
    final_table = f"""════════════════════════════════════════════════════
FINAL THREE-WAY COMPARISON
════════════════════════════════════════════════════
Method                        Val F1    Test F1
────────────────────────────────────────────────
MFCC + Random Forest           0.81      0.57
YAMNet + SVM (default)         {baseline_val_f1:.2f}      {baseline_test_f1:.2f}
YAMNet + SVM (optimized)       {opt_val_f1:.2f}      {opt_test_f1:.2f}  ← new
────────────────────────────────────────────────
Best overall: {winner}
════════════════════════════════════════════════════"""
    print(final_table)
    
    # Step 8 — Save optimized model
    print("\nStep 8: Saving optimized model...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(opt_svm, MODEL_SAVE_PATH)
    print(f"Optimized model saved to {MODEL_SAVE_PATH}")

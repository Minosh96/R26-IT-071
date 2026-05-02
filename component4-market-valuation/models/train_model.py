import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os
import math
import sys

# Ensure terminal output handles Unicode characters
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback for older python versions if needed
        pass
DATA_PATH = "data/processed/alto_augmented.csv"
MODEL_SAVE_PATH = "models/saved/price_model.joblib"
SCALER_SAVE_PATH = "models/saved/price_scaler.joblib"
FEATURE_NAMES_PATH = "models/saved/feature_names.json"
RANDOM_SEED = 42
FEATURES = [
    'maf_year', 'vehicle_age', 'mileage_km', 'previous_owners',
    'is_reconditioned', 'power_shutters', 'power_mirrors', 'reg_gap'
]
TARGET = 'price_million'

def train_model():
    # Step 1 — Load and validate data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please run augmentation script first.")
    
    df = pd.read_csv(DATA_PATH)
    
    # Drop data_source if it exists
    real_count = 0
    synthetic_count = 0
    if 'data_source' in df.columns:
        real_count = len(df[df['data_source'] == 'real'])
        synthetic_count = len(df[df['data_source'] == 'synthetic'])
        df = df.drop(columns=['data_source'])
    
    # Validate features
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features in dataset: {missing_features}")
    
    # Confirm no nulls
    for col in FEATURES + [TARGET]:
        if df[col].isnull().any():
            raise ValueError(f"Column '{col}' contains null values.")
    
    print(f"Total rows loaded: {len(df)}")
    if real_count or synthetic_count:
        print(f"Data composition: {real_count} real + {synthetic_count} synthetic")
    print("Validation: No nulls found in features or target.")

    # Step 2 — Split data
    X = df[FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )
    print(f"Data split: Train={len(X_train)}, Test={len(X_test)}")

    # Step 3 — Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Fitted scaler saved to {SCALER_SAVE_PATH}")

    # Step 4 — Train three models and compare
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=None,
            min_samples_split=5, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1,
            max_depth=4, random_state=RANDOM_SEED
        ),
        "Linear Regression": LinearRegression()
    }
    
    results = {}
    
    print("\n─────────────────────────────────────────────────────────────")
    print(f"{'Model':<20} {'MAE (LKR)':<15} {'RMSE (M)':<12} {'R²':<10}")
    print("─────────────────────────────────────────────────────────────")
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        mae_lkr = mae * 1_000_000
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mae': mae,
            'mae_lkr': mae_lkr,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"{name:<20} {mae_lkr:,.0f}{'':<8} {rmse:<12.2f} {r2:<10.2f}")
    
    print("─────────────────────────────────────────────────────────────")

    # Step 5 — Select best model
    best_model_name = min(results, key=lambda x: results[x]['mae'])
    best_result = results[best_model_name]
    print(f"\nBest model: {best_model_name}")
    print(f"Metrics: MAE={best_result['mae_lkr']:,.0f} LKR, RMSE={best_result['rmse']:.2f}M, R²={best_result['r2']:.2f}")

    # Step 6 — Cross validation on best model
    X_scaled = scaler.transform(X) # Scale full dataset for CV
    cv_scores = cross_val_score(
        best_result['model'], X_scaled, y, 
        cv=5, scoring='neg_mean_absolute_error'
    )
    cv_mae_mean = -cv_scores.mean()
    cv_mae_std = cv_scores.std()
    print(f"Cross-validation MAE: {cv_mae_mean:.4f} ± {cv_mae_std:.4f} million LKR")

    # Step 7 — Feature importance
    print("\nFeature importance ranking:")
    if hasattr(best_result['model'], 'feature_importances_'):
        importances = best_result['model'].feature_importances_
        feature_importance = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
        
        for i, (feat, imp) in enumerate(feature_importance, 1):
            bar = '█' * int(imp * 50)
            print(f"  {i}. {feat:<18} {imp:.2f}  {bar}")
    else:
        print("  (Feature importance not available for this model type)")

    # Step 8 — Prediction examples
    print("\nPrediction sanity check:")
    # Sample 1: 2015 Alto, 80000km, 2nd owner, BN, shutters yes, mirrors no, reg_gap 0
    # Sample 2: 2014 Alto, 120000km, 3rd owner, BN, shutters yes, mirrors no, reg_gap 0
    # Sample 3: 2013 Alto, 150000km, 4th owner, BN, shutters no, mirrors no, reg_gap 1
    
    current_year = 2026
    samples = [
        [2015, current_year-2015, 80000, 2, 0, 1, 0, 0],
        [2014, current_year-2014, 120000, 3, 0, 1, 0, 0],
        [2013, current_year-2013, 150000, 4, 0, 0, 0, 1]
    ]
    
    samples_df = pd.DataFrame(samples, columns=FEATURES)
    samples_scaled = scaler.transform(samples_df)
    sample_preds = best_result['model'].predict(samples_scaled)
    
    sample_desc = [
        "2015, 80000km, 2nd owner",
        "2014, 120000km, 3rd owner",
        "2013, 150000km, 4th owner"
    ]
    
    for desc, pred in zip(sample_desc, sample_preds):
        print(f"  {desc:<25} → LKR {pred * 1_000_000:,.0f}")

    # Step 9 — Save best model
    joblib.dump(best_result['model'], MODEL_SAVE_PATH)
    with open(FEATURE_NAMES_PATH, 'w') as f:
        json.dump(FEATURES, f)
    print(f"\nModel saved: {MODEL_SAVE_PATH}")
    print(f"Feature names saved: {FEATURE_NAMES_PATH}")

    # Step 10 — Print final summary
    print("\n" + "═" * 48)
    print("MODEL TRAINING SUMMARY")
    print("═" * 48)
    print(f"{'Dataset:':<17} {len(df)} records ({real_count} real + {synthetic_count} synthetic)")
    print(f"{'Best model:':<17} {best_model_name}")
    print(f"{'MAE:':<17} LKR {best_result['mae_lkr']:,.0f}")
    print(f"{'RMSE:':<17} {best_result['rmse']:.2f} million LKR")
    print(f"{'R² score:':<17} {best_result['r2']:.2f}")
    print(f"{'CV MAE (5-fold):':<17} {cv_mae_mean:.2f} ± {cv_mae_std:.2f} million LKR")
    
    print("\nTop price factors:")
    if hasattr(best_result['model'], 'feature_importances_'):
        for i, (feat, imp) in enumerate(feature_importance[:3], 1):
            print(f"  {i}. {feat.replace('_', ' ').capitalize():<15} ({int(imp*100)}%)")
    
    print("\nPrediction sanity check:")
    for desc, pred in zip(sample_desc, sample_preds):
        print(f"  {desc:<25} → LKR {pred * 1_000_000:,.0f}")
    
    print(f"\nModel saved: {MODEL_SAVE_PATH}")
    print(f"Scaler saved: {SCALER_SAVE_PATH}")
    print("═" * 48)

if __name__ == "__main__":
    try:
        train_model()
        print("\nModel training complete")
    except Exception as e:
        print(f"Error during training: {e}")

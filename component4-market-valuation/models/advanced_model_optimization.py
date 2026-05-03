import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os
import time
import sys

# Ensure UTF-8 output for Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Configuration
DATA_PATH = "data/processed/alto_augmented.csv"
RESULTS_SAVE_PATH = "models/saved/ensemble_results.json"
BEST_MODEL_PATH = "models/saved/advanced_best_model.joblib"
BEST_SCALER_PATH = "models/saved/advanced_scaler.joblib"
FEATURE_NAMES_PATH = "models/saved/feature_names.json"
RANDOM_SEED = 42
FEATURES = [
    'maf_year', 'vehicle_age', 'mileage_km', 'previous_owners',
    'is_reconditioned', 'power_shutters', 'power_mirrors', 'reg_gap'
]
TARGET = 'price_million'

def run_experiment():
    print("--- Starting Advanced Model Optimization (Ensemble Experiment) ---")
    
    # Step 1 — Load and prepare data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    if 'data_source' in df.columns:
        df = df.drop(columns=['data_source'])
    
    X = df[FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dataset loaded: {len(df)} records")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Step 2 — Define base models
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None,
        min_samples_split=5, 
        random_state=RANDOM_SEED, n_jobs=-1
    )
    
    gb = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=5, random_state=RANDOM_SEED
    )
    
    lr = LinearRegression()

    # Step 3 — Define Ensemble Combinations
    experiments = [
        {"name": "Random Forest (RF)", "model": rf, "short": "RF"},
        {"name": "Gradient Boosting (GB)", "model": gb, "short": "GB"},
        {"name": "Linear Regression (LR)", "model": lr, "short": "LR"},
        {
            "name": "RF + GB", 
            "short": "RF + GB",
            "model": VotingRegressor([('rf', rf), ('gb', gb)])
        },
        {
            "name": "RF + LR", 
            "short": "RF + LR",
            "model": VotingRegressor([('rf', rf), ('lr', lr)])
        },
        {
            "name": "GB + LR", 
            "short": "GB + LR",
            "model": VotingRegressor([('gb', gb), ('lr', lr)])
        },
        {
            "name": "RF + GB + LR (all three)", 
            "short": "RF + GB + LR",
            "model": VotingRegressor([('rf', rf), ('gb', gb), ('lr', lr)])
        }
    ]

    results = []
    
    print("\n" + "─"*70)
    print(f"{'Experiment':<30} | {'MAE (LKR)':<12} | {'RMSE':<8} | {'R²':<6}")
    print("─"*70)

    for i, exp in enumerate(experiments, 1):
        start_time = time.time()
        model = exp["model"]
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        duration = time.time() - start_time
        
        mae_lkr = mae * 1_000_000
        
        results.append({
            "id": i,
            "name": exp["name"],
            "short_name": exp["short"],
            "mae_lkr": mae_lkr,
            "mae_mil": mae,
            "rmse": rmse,
            "r2": r2,
            "time": duration,
            "model_obj": model
        })
        
        print(f"{exp['name']:<30} | {mae_lkr:,.0f}{'':<4} | {rmse:<8.3f} | {r2:<6.3f}")

    print("─"*70)

    # Step 4 — Find Winner and Save
    winner = min(results, key=lambda x: x["mae_mil"])
    for res in results:
        res["is_winner"] = (res["id"] == winner["id"])
    
    # Cross-validation for the winner
    print(f"\nWinner: {winner['name']}")
    print("Performing 5-fold cross-validation on winner...")
    
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(winner["model_obj"], scaler.transform(X), y, cv=cv, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"CV MAE: {cv_mae:.3f} ± {cv_std:.3f} million LKR")

    # Save Best Model and Scaler
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
    joblib.dump(winner["model_obj"], BEST_MODEL_PATH)
    joblib.dump(scaler, BEST_SCALER_PATH)
    
    # Save results (excluding objects)
    json_results = []
    for r in results:
        r_copy = r.copy()
        del r_copy["model_obj"]
        json_results.append(r_copy)
        
    with open(RESULTS_SAVE_PATH, 'w') as f:
        json.dump(json_results, f, indent=4)
        
    print(f"\nSaved best model to: {BEST_MODEL_PATH}")
    print(f"Saved results to: {RESULTS_SAVE_PATH}")
    print("--- Experiment Complete ---")

if __name__ == "__main__":
    run_experiment()

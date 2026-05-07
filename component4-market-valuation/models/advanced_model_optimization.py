import pandas as pd
import numpy as np
import os
import joblib
import json
import time
import sys
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Ensure UTF-8 output for Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# CONFIGURATION
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


def main():
    print("=== STEP 1: LOAD DATA ===")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if 'data_source' in df.columns:
        df = df.drop(columns=['data_source'])
        print("Dropped 'data_source' column.")

    print("=== STEP 2: FEATURE ENGINEERING ===")
    df['mileage_per_year'] = df['mileage_km'] / (df['vehicle_age'] + 1)
    df['age_squared'] = df['vehicle_age'] ** 2
    df['mileage_squared'] = df['mileage_km'] ** 2
    df['owner_age_interaction'] = df['previous_owners'] * df['vehicle_age']

    extended_features = FEATURES + ['mileage_per_year', 'age_squared', 'mileage_squared', 'owner_age_interaction']
    print(f"Extended features: {extended_features}")

    X = df[extended_features]
    y = df[TARGET]

    print("Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("=== STEP 3: DEFINE MODELS ===")
    models = {
        'RandomForest': RandomForestRegressor(random_state=RANDOM_SEED),
        'GradientBoosting': GradientBoostingRegressor(random_state=RANDOM_SEED),
        'XGBoost': XGBRegressor(random_state=RANDOM_SEED, objective='reg:squarederror'),
        'LightGBM': LGBMRegressor(random_state=RANDOM_SEED)
    }

    print("=== STEP 4: HYPERPARAMETER TUNING ===")
    param_grids = {
        'RandomForest': {
            'n_estimators': [200, 300, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5]
        },
        'XGBoost': {
            'n_estimators': [200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        },
        'LightGBM': {
            'n_estimators': [200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100]
        }
    }

    best_models = {}
    best_score = float('inf')  # Optimizing for MAE, so lower is better
    overall_best_model = None
    overall_best_name = ""

    print("Starting RandomizedSearchCV for each model...")
    for name, model in models.items():
        print(f"\n--- Tuning {name} ---")
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grids[name],
            n_iter=10,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbose=1
        )

        print("=== STEP 5: TRAIN BEST MODELS ===")
        random_search.fit(X_train_scaled, y_train)

        best_model = random_search.best_estimator_
        best_models[name] = best_model

        print(f"Best parameters for {name}: {random_search.best_params_}")

        # Evaluate on test set
        y_pred = best_model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        print(f"{name} Test Performance:")
        print(f"  R2 Score: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")

        # Track overall best model based on MAE
        if mae < best_score:
            best_score = mae
            overall_best_model = best_model
            overall_best_name = name

    print(f"\n=== OVERALL BEST MODEL ===")
    print(f"Selected: {overall_best_name} with MAE: {best_score:.4f}")

    print("\nSaving best model and scaler...")
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
    joblib.dump(overall_best_model, BEST_MODEL_PATH)
    joblib.dump(scaler, BEST_SCALER_PATH)
    print(f"Saved model to: {BEST_MODEL_PATH}")
    print(f"Saved scaler to: {BEST_SCALER_PATH}")
    print("Optimization Complete.")


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
    main()
    run_experiment()

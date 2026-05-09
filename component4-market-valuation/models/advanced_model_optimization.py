import pandas as pd
import numpy as np
import os
import joblib
import json
import time
import sys
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
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
DATA_PATH = "data/processed/holistic_augmented.csv"
RESULTS_SAVE_PATH = "models/saved/ensemble_results.json"
BEST_MODEL_PATH = "models/saved/advanced_best_model.joblib"
BEST_SCALER_PATH = "models/saved/advanced_scaler.joblib"
FEATURE_NAMES_PATH = "models/saved/advanced_feature_names.json"
RANDOM_SEED = 42

# We now include the inter-component features
FEATURES = [
    'maf_year', 'vehicle_age', 'mileage_km', 'previous_owners',
    'is_reconditioned', 'power_shutters', 'power_mirrors', 'reg_gap',
    'fault_class', 'confidence', 'body_score'
]

TARGET = 'price_million'

def get_extended_features(df):
    df_new = df.copy()
    
    # Feature Selection: robust derived features
    df_new['mileage_per_year'] = df_new['mileage_km'] / (df_new['vehicle_age'] + 1)
    df_new['log_mileage_km'] = np.log1p(df_new['mileage_km'])
    df_new['log_vehicle_age'] = np.log1p(df_new['vehicle_age'])
    
    extended_cols = FEATURES + [
        'mileage_per_year', 'log_mileage_km', 'log_vehicle_age'
    ]
    return df_new, extended_cols

def run_optimization():
    print("--- Starting Holistic Model Optimization (Phase 3: Integration) ---")

    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    if 'data_source' in df.columns:
        df = df.drop(columns=['data_source'])

    print("=== STEP 1: FEATURE ENGINEERING & PREPROCESSING ===")
    df, extended_features = get_extended_features(df)
    print(f"Total features used: {len(extended_features)}")

    X = df[extended_features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )

    # Separate numeric and categorical columns
    categorical_cols = ['fault_class']
    numeric_cols = [c for c in extended_features if c not in categorical_cols]

    # Preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Fit transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Processed feature matrix shape: {X_train_processed.shape}")

    # Step 2 — Define Base Models and Parameter Grids
    print("\n=== STEP 2: HYPERPARAMETER TUNING BASE MODELS ===")
    
    models_to_tune = {
        'rf': (RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1), {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }),
        'gb': (GradientBoostingRegressor(random_state=RANDOM_SEED), {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5]
        }),
        'xgb': (XGBRegressor(random_state=RANDOM_SEED, objective='reg:squarederror'), {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }),
        'lgbm': (LGBMRegressor(random_state=RANDOM_SEED, verbose=-1), {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100]
        })
    }

    best_estimators = []
    
    start_time = time.time()
    for name, (model, param_grid) in models_to_tune.items():
        print(f"  -> Tuning {name}...")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=RANDOM_SEED
        )
        search.fit(X_train_processed, y_train)
        best_estimators.append((name, search.best_estimator_))
        print(f"     Best params: {search.best_params_}")

    print("\n=== STEP 3: TRAINING FINAL STACKING ENSEMBLE ===")
    
    stacking_regressor = StackingRegressor(
        estimators=best_estimators,
        final_estimator=Ridge(),
        cv=5,
        n_jobs=-1
    )
    
    # We create a final pipeline that bundles the preprocessor and the stacked model
    # This allows `inference/valuate.py` to pass the raw dataframe directly!
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', stacking_regressor)
    ])
    
    final_pipeline.fit(X_train, y_train) # Fit pipeline on raw data
    y_pred = final_pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_test - y_pred) / y_test))
    accuracy_percent = (1 - mape) * 100
    
    duration = time.time() - start_time

    mae_lkr = mae * 1_000_000

    print(f"\nModel Performance (Test Set):")
    print(f"  MAE: LKR {mae_lkr:,.0f} ({mae:.4f} M)")
    print(f"  MAPE: {mape:.2%}")
    print(f"  Prediction Accuracy: {accuracy_percent:.2f}%")
    print(f"  RMSE: {rmse:.4f} M")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  Total Optimization Time: {duration:.2f}s")

    # Cross-validation for the winner
    print("\nPerforming 5-fold cross-validation...")
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(final_pipeline, X, y, cv=cv, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()

    print(f"CV MAE: {cv_mae:.3f} ± {cv_std:.3f} million LKR")

    # Save Best Model Pipeline
    print("\n=== STEP 4: SAVING ASSETS ===")
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
    joblib.dump(final_pipeline, BEST_MODEL_PATH)
    # Note: We don't save BEST_SCALER_PATH anymore since the scaler is built INTO the pipeline!
    
    with open(FEATURE_NAMES_PATH, 'w') as f:
        json.dump(extended_features, f)

    result_dict = {
        "name": "Holistic Stacking Ensemble Pipeline",
        "mae_lkr": mae_lkr,
        "mae_mil": mae,
        "mape": mape,
        "accuracy_percent": accuracy_percent,
        "rmse": rmse,
        "r2": r2,
        "cv_mae": cv_mae,
        "cv_std": cv_std,
        "time": duration
    }
    
    with open(RESULTS_SAVE_PATH, 'w') as f:
        json.dump([result_dict], f, indent=4)

    print(f"Saved best model pipeline to: {BEST_MODEL_PATH}")
    print("--- Holistic Integration Complete ---")

if __name__ == "__main__":
    run_optimization()

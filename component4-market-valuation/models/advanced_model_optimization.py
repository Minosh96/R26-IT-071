import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# CONFIGURATION
DATA_PATH = "data/processed/alto_augmented.csv"
BEST_MODEL_PATH = "models/saved/advanced_best_model.joblib"
BEST_SCALER_PATH = "models/saved/advanced_scaler.joblib"
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
    best_score = float('inf') # We are optimizing for MAE, so lower is better
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
        
        # Track overall best model based on MAE (could also use R2)
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

if __name__ == "__main__":
    main()

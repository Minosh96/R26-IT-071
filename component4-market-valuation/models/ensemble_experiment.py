import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os
import time
import sys

# Ensure UTF-8 output for Windows console
sys.stdout.reconfigure(encoding='utf-8')

DATA_PATH = "data/processed/alto_augmented.csv"
RESULTS_SAVE_PATH = "models/saved/ensemble_results.json"
BEST_MODEL_PATH = "models/saved/best_ensemble_model.joblib"
BEST_SCALER_PATH = "models/saved/best_ensemble_scaler.joblib"
RANDOM_SEED = 42
FEATURES = [
    'maf_year', 'vehicle_age', 'mileage_km', 'previous_owners',
    'is_reconditioned', 'power_shutters', 'power_mirrors', 'reg_gap'
]
TARGET = 'price_million'

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(os.path.dirname(RESULTS_SAVE_PATH), exist_ok=True)

    print("Step 1 — Load and prepare data")
    df = pd.read_csv(DATA_PATH)
    if 'data_source' in df.columns:
        df = df.drop(columns=['data_source'])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=None, random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    print("\nStep 2 — Define base models")
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None,
        min_samples_split=5,
        random_state=RANDOM_SEED, n_jobs=-1
    )

    gb = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1,
        max_depth=4, random_state=RANDOM_SEED
    )

    lr = LinearRegression()

    print("\nStep 3 — Run all 7 experiments")
    experiments = [
        ("Random Forest (RF)", rf, "RF"),
        ("Gradient Boosting (GB)", gb, "GB"),
        ("Linear Regression (LR)", lr, "LR"),
        ("RF + GB", VotingRegressor(estimators=[('rf', rf), ('gb', gb)]), "RF + GB"),
        ("RF + LR", VotingRegressor(estimators=[('rf', rf), ('lr', lr)]), "RF + LR"),
        ("GB + LR", VotingRegressor(estimators=[('gb', gb), ('lr', lr)]), "GB + LR"),
        ("RF + GB + LR (all three)", VotingRegressor(estimators=[('rf', rf), ('gb', gb), ('lr', lr)]), "RF + GB + LR")
    ]

    results = []
    
    for name, model, short_name in experiments:
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        mae_lkr = mae * 1_000_000
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "id": len(results) + 1,
            "name": name,
            "short_name": short_name,
            "model": model,
            "mae_lkr": mae_lkr,
            "mae_mil": mae,
            "rmse": rmse,
            "r2": r2,
            "time": train_time
        })

    # Find winner
    winner = min(results, key=lambda x: x['mae_lkr'])
    
    # Base GB model for comparison
    gb_result = next(r for r in results if r['short_name'] == 'GB')

    print("\nStep 4 — Print comparison table")
    print("══════════════════════════════════════════════════════════════════")
    print("ENSEMBLE EXPERIMENT RESULTS — Suzuki Alto Price Prediction")
    print("══════════════════════════════════════════════════════════════════")
    print(f"{'#':<3} {'Model Combination':<36} {'MAE(LKR)':<10} {'RMSE(M)':<8} {'R²':<5} {'Time(s)':<7}")
    print("──────────────────────────────────────────────────────────────────")
    for r in results:
        print(f"{r['id']:<3} {r['name']:<36} {r['mae_lkr']:<10,.0f} {r['rmse']:<8.2f} {r['r2']:<5.2f} {r['time']:<7.1f}")
    print("──────────────────────────────────────────────────────────────────")
    print(f"Winner: {winner['name']} with MAE LKR {winner['mae_lkr']:,.0f}")
    print("══════════════════════════════════════════════════════════════════")

    print("\nStep 5 — Cross validate the best model")
    # For CV, we need to scale inside the fold to prevent data leakage.
    cv_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', winner['model'])
    ])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(cv_pipeline, X, y, cv=kf, scoring='neg_mean_absolute_error')
    cv_mae_scores = -cv_scores
    cv_mean_mae = cv_mae_scores.mean()
    cv_std_mae = cv_mae_scores.std()
    print(f"5-Fold CV Mean MAE: {cv_mean_mae:.2f} million LKR (±{cv_std_mae:.2f})")

    print("\nStep 6 — Feature importance for best model")
    best_model = winner['model']
    importances = None
    
    if winner['short_name'] in ['RF', 'GB']:
        importances = best_model.feature_importances_
    elif isinstance(best_model, VotingRegressor):
        tree_importances = []
        for name, est in best_model.named_estimators_.items():
            if hasattr(est, 'feature_importances_'):
                tree_importances.append(est.feature_importances_)
        if tree_importances:
            importances = np.mean(tree_importances, axis=0)
            
    if importances is not None:
        feat_imp = pd.DataFrame({'Feature': FEATURES, 'Importance': importances})
        feat_imp = feat_imp.sort_values('Importance', ascending=False)
        print(feat_imp.to_string(index=False))
    else:
        print("Feature importance not applicable for the winning model.")

    print("\nStep 7 — Prediction sanity check on winning model")
    # Sample 1: 2015 Alto, 80000km, 2nd owner, BN, shutters yes, mirrors no
    # Sample 2: 2014 Alto, 120000km, 3rd owner, BN, shutters yes, mirrors no
    # Sample 3: 2013 Alto, 150000km, 4th owner, BN, shutters no, mirrors no
    # Assuming current year for vehicle age calculation is around 2024 (9, 10, 11 years old respectively)
    samples = pd.DataFrame([
        {'maf_year': 2015, 'vehicle_age': 9, 'mileage_km': 80000, 'previous_owners': 2, 'is_reconditioned': 0, 'power_shutters': 1, 'power_mirrors': 0, 'reg_gap': 0},
        {'maf_year': 2014, 'vehicle_age': 10, 'mileage_km': 120000, 'previous_owners': 3, 'is_reconditioned': 0, 'power_shutters': 1, 'power_mirrors': 0, 'reg_gap': 0},
        {'maf_year': 2013, 'vehicle_age': 11, 'mileage_km': 150000, 'previous_owners': 4, 'is_reconditioned': 0, 'power_shutters': 0, 'power_mirrors': 0, 'reg_gap': 0}
    ], columns=FEATURES)
    
    samples_scaled = scaler.transform(samples)
    predictions = best_model.predict(samples_scaled)
    for i, pred in enumerate(predictions):
        print(f"Sample {i+1} prediction: {pred:.2f} million LKR (LKR {pred*1_000_000:,.0f})")

    print("\nStep 8 — Save results and best model")
    save_results = []
    for r in results:
        res = r.copy()
        del res['model']
        res['is_winner'] = (res['id'] == winner['id'])
        save_results.append(res)
        
    with open(RESULTS_SAVE_PATH, 'w') as f:
        json.dump(save_results, f, indent=4)
        
    joblib.dump(winner['model'], BEST_MODEL_PATH)
    joblib.dump(scaler, BEST_SCALER_PATH)
    print("Files saved successfully.")

    print("\nStep 9 — Print final summary")
    print("════════════════════════════════════════════════════════")
    print("FINAL SUMMARY")
    print("════════════════════════════════════════════════════════")
    # For reporting dataset size we can approximate based on user prompt '1000 records (338 real + 662 synthetic)' if we don't know the exact count
    print(f"Dataset:          {len(df)} records")
    print("Experiments run:  7 (3 individual + 3 pairs + 1 all)")
    print(f"Best combination: {winner['short_name']}")
    print(f"Best MAE:         LKR {winner['mae_lkr']:,.0f}")
    print(f"Best R²:          {winner['r2']:.2f}")
    print(f"CV MAE (5-fold):  {cv_mean_mae:.2f} ± {cv_std_mae:.2f} million LKR")
    print("")
    print("Improvement over single best model (GB alone):")
    mae_diff = gb_result['mae_lkr'] - winner['mae_lkr']
    r2_diff = winner['r2'] - gb_result['r2']
    
    if mae_diff > 0:
        print(f"  MAE reduced by:  LKR {mae_diff:,.0f}")
    else:
        print(f"  MAE increased by:  LKR {-mae_diff:,.0f}")
        
    if r2_diff > 0:
        print(f"  R² improved by:  +{r2_diff:.2f}")
    else:
        print(f"  R² worsened by:  {r2_diff:.2f}")
        
    print("")
    print(f"Best model saved: {BEST_MODEL_PATH}")
    print("════════════════════════════════════════════════════════")

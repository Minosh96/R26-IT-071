"""
Train and evaluate the market price model on REAL listings only.

Evaluation (all numbers are on real prices that the model never trained on):
  H) 20% real hold-out       - set aside before anything else, never trained on
  A) Grouped 5-fold CV       - the same car never sits in both train and test
  B) Out-of-time test        - train on Jan-Mar 2026 listings, test on Oct 2026
  C) Recalibration test      - B plus a small batch of fresh Oct listings,
                               simulating the monthly retraining procedure
Every model is compared with simple baselines (year average, linear regression).

Outputs:
  models/saved/market_model.joblib       final model fitted on all real listings
  models/saved/market_model_meta.json    features, price interval, scope
  models/saved/market_model_report.json  all evaluation numbers
  thesis/figures/v2_*.png                figures for the presentation
"""
import os
import sys
import json
import pathlib
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
from models.market_model import FEATURES, HybridRegressor, add_features

warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/alto_real.csv"
SAVE_DIR = "models/saved"
FIG_DIR = "thesis/figures"
SEED = 42
INTERVAL = 0.80          # fair price range covers 80% of real listings
OUT_OF_TIME_MONTH = 10   # Oct 2026 listings form the future test set
FRESH_LISTINGS = 40      # listings collected per monthly retrain in test C


class YearMean:
    """Baseline: the average listed price for that manufacture year."""

    def fit(self, X, y):
        self.means_ = y.groupby(X["maf_year"]).mean()
        self.overall_ = y.mean()
        return self

    def predict(self, X):
        return X["maf_year"].map(self.means_).fillna(self.overall_).values


CANDIDATES = {
    "Year-average baseline": lambda: YearMean(),
    "Linear (Ridge) regression": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
    "Random Forest": lambda: RandomForestRegressor(400, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=SEED),
    "XGBoost": lambda: XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=SEED),
    "Hybrid (constrained linear + boosting)": lambda: HybridRegressor(random_state=SEED),
}
CHOSEN = "Hybrid (constrained linear + boosting)"


def metrics(y_true, y_pred):
    err = np.abs(y_true - y_pred)
    return {
        "mae_lkr": float(err.mean()),
        "mape_pct": float(100 * (err / y_true).mean()),
        "r2": float(1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()),
        "within_5_pct": float(100 * (err / y_true <= 0.05).mean()),
        "n": int(len(y_true)),
    }


def print_table(title, results):
    print(f"\n{title}")
    print(f"  {'model':28s} {'MAE (LKR)':>10s} {'MAPE':>7s} {'R2':>6s} {'within 5%':>10s}")
    for name, m in results.items():
        print(f"  {name:28s} {m['mae_lkr']:10,.0f} {m['mape_pct']:6.2f}% {m['r2']:6.3f} {m['within_5_pct']:9.1f}%")


def main():
    df = pd.read_csv(DATA_PATH)
    X = add_features(df)
    y = df["price_million"] * 1_000_000
    groups = df["group"].values
    print(f"Real listings: {len(df)} (manufactured {df.maf_year.min()}-{df.maf_year.max()})")

    report = {"data": {"rows": len(df), "year_range": [int(df.maf_year.min()), int(df.maf_year.max())]}}

    # H) 20% of real listings held out (grouped by car), trained on the other 80%
    train_idx, hold_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED).split(X, y, groups))
    report["holdout_20pct"] = {}
    for name, make in CANDIDATES.items():
        pred = make().fit(X.iloc[train_idx], y.iloc[train_idx]).predict(X.iloc[hold_idx])
        report["holdout_20pct"][name] = metrics(y.iloc[hold_idx].values, pred)
    print_table(f"H) 20% real hold-out (train n={len(train_idx)}, test n={len(hold_idx)})", report["holdout_20pct"])

    # A) Grouped 5-fold cross-validation
    oof = {name: np.zeros(len(df)) for name in CANDIDATES}
    for train_idx, test_idx in GroupKFold(5).split(X, y, groups):
        for name, make in CANDIDATES.items():
            model = make().fit(X.iloc[train_idx], y.iloc[train_idx])
            oof[name][test_idx] = model.predict(X.iloc[test_idx])
    report["grouped_cv"] = {name: metrics(y.values, pred) for name, pred in oof.items()}
    print_table("A) Grouped 5-fold CV, real listings only", report["grouped_cv"])

    # Price interval: empirical (conformal) quantiles of the out-of-fold errors
    residuals = y.values - oof[CHOSEN]
    lo_q, hi_q = np.quantile(residuals, [(1 - INTERVAL) / 2, 1 - (1 - INTERVAL) / 2])
    ratio = residuals / oof[CHOSEN]
    lo_r, hi_r = np.quantile(ratio, [(1 - INTERVAL) / 2, 1 - (1 - INTERVAL) / 2])
    covered = ((y.values >= oof[CHOSEN] * (1 + lo_r)) & (y.values <= oof[CHOSEN] * (1 + hi_r))).mean()
    report["price_interval"] = {"level": INTERVAL, "low_ratio": float(lo_r), "high_ratio": float(hi_r),
                                "low_lkr": float(lo_q), "high_lkr": float(hi_q), "cv_coverage": float(covered)}
    print(f"\n  {int(INTERVAL*100)}% fair price range = prediction {lo_r:+.1%} / {hi_r:+.1%} (coverage {covered:.1%})")

    # B) Out-of-time test and C) monthly recalibration
    past = (df["listed_months"] < OUT_OF_TIME_MONTH).values
    future = ~past
    report["out_of_time"] = {}
    for name, make in CANDIDATES.items():
        pred = make().fit(X[past], y[past]).predict(X[future])
        report["out_of_time"][name] = metrics(y[future].values, pred)
    print_table(f"B) Trained on listings before Oct 2026 (n={past.sum()}), tested on Oct 2026 (n={future.sum()})",
                report["out_of_time"])

    rng = np.random.default_rng(SEED)
    future_idx = np.where(future)[0]
    recal = {name: [] for name in CANDIDATES}
    for _ in range(10):
        fresh = rng.choice(future_idx, FRESH_LISTINGS, replace=False)
        train_mask, test_mask = past.copy(), future.copy()
        train_mask[fresh], test_mask[fresh] = True, False
        for name, make in CANDIDATES.items():
            pred = make().fit(X[train_mask], y[train_mask]).predict(X[test_mask])
            recal[name].append(metrics(y[test_mask].values, pred))
    report["recalibrated"] = {name: {k: float(np.mean([r[k] for r in runs])) for k in runs[0]} for name, runs in recal.items()}
    print_table(f"C) Same as B, plus {FRESH_LISTINGS} fresh Oct listings added to training (10 repeats)",
                report["recalibrated"])

    month_prices = df.groupby("listed_months")["price_million"].agg(["count", "mean", "median"])
    report["price_by_listing_month"] = {int(k): {c: float(v) for c, v in row.items()} for k, row in month_prices.iterrows()}

    # Final model on all real listings
    final = CANDIDATES[CHOSEN]().fit(X, y)
    pi = permutation_importance(final, X, y, n_repeats=20, random_state=SEED, scoring="neg_mean_absolute_error")
    report["permutation_importance_lkr"] = {f: float(v) for f, v in sorted(zip(FEATURES, pi.importances_mean), key=lambda t: -t[1])}

    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(final, f"{SAVE_DIR}/market_model.joblib")
    meta = {
        "model": CHOSEN,
        "features": FEATURES,
        "training_rows": len(df),
        "year_range": report["data"]["year_range"],
        "market_month": int(df["listed_months"].max()),
        "price_interval": report["price_interval"],
        "cv_mae_lkr": report["grouped_cv"][CHOSEN]["mae_lkr"],
        "cv_mape_pct": report["grouped_cv"][CHOSEN]["mape_pct"],
        "holdout_mae_lkr": report["holdout_20pct"][CHOSEN]["mae_lkr"],
        "holdout_mape_pct": report["holdout_20pct"][CHOSEN]["mape_pct"],
        # Inputs outside these ranges are refused (the model has never seen such cars)
        "scope": {
            "make": "Suzuki",
            "model": "Alto",
            "mileage_km": [int(df["mileage_km"].min()), int(df["mileage_km"].max())],
            "previous_owners": [int(df["previous_owners"].min()), int(df["previous_owners"].max())],
        },
        # A typical listing, used to explain each price ("-LKR 45,000 for higher mileage")
        "reference_vehicle": {
            "maf_year": int(df["maf_year"].median()),
            "reg_gap": int((df["reg_year"] - df["maf_year"]).clip(lower=0).median()),
            "mileage_km": int(round(df["mileage_km"].median(), -3)),
            "previous_owners": int(df["previous_owners"].median()),
            "is_reconditioned": int(df["is_reconditioned"].mode()[0]),
            "power_shutters": int(df["power_shutters"].mode()[0]),
            "power_mirrors": int(df["power_mirrors"].mode()[0]),
        },
    }
    with open(f"{SAVE_DIR}/market_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(f"{SAVE_DIR}/market_model_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {SAVE_DIR}/market_model.joblib, market_model_meta.json, market_model_report.json")

    make_figures(df, y, oof[CHOSEN], report)


def make_figures(df, y, pred, report):
    os.makedirs(FIG_DIR, exist_ok=True)

    names = list(report["grouped_cv"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, [report["grouped_cv"][n]["mae_lkr"] / 1000 for n in names],
            color=["#999" if n == "Year-average baseline" else "#2a6fdb" if n == CHOSEN else "#8fb3ee" for n in names])
    ax.set_xlabel("Average error on unseen real listings (LKR thousand)")
    ax.set_title("Grouped 5-fold CV, real listings only")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/v2_model_vs_baseline.png", dpi=200); plt.close(fig)

    imp = report["permutation_importance_lkr"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(list(imp), [v / 1000 for v in imp.values()], color="#2a6fdb")
    ax.set_xlabel("Increase in error when the factor is shuffled (LKR thousand)")
    ax.set_title("What drives the market price")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/v2_price_drivers.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(pred / 1e6, y / 1e6, s=10, alpha=0.5, color="#2a6fdb")
    lim = [min(pred.min(), y.min()) / 1e6, max(pred.max(), y.max()) / 1e6]
    ax.plot(lim, lim, color="#333", lw=1)
    ax.set_xlabel("Predicted price (LKR Mn)"); ax.set_ylabel("Real listed price (LKR Mn)")
    ax.set_title("Predicted vs real (out-of-fold)")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/v2_pred_vs_real.png", dpi=200); plt.close(fig)

    recent = df[df["listed_months"] >= 1]
    by_month = recent.groupby("listed_months")["price_million"].median()
    labels = {1: "Jan", 2: "Feb", 3: "Mar", 10: "Oct"}
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar([labels.get(m, str(m)) for m in by_month.index], by_month.values, color="#2a6fdb")
    ax.set_ylim(by_month.min() * 0.95, by_month.max() * 1.02)
    ax.set_ylabel("Median listed price (LKR Mn)")
    ax.set_title("Alto asking prices drift over 2026")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/v2_price_drift.png", dpi=200); plt.close(fig)
    print(f"Saved figures to {FIG_DIR}/v2_*.png")


if __name__ == "__main__":
    main()

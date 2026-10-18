"""
Evaluation script for the Market Value Analysis component (Component 4).

Run from the component root:
    venv/Scripts/python thesis/analysis/evaluate_component.py

It does not modify the trained model or the datasets. It reads them, repeats the
train/test split used in models/advanced_model_optimization.py, and writes
    thesis/analysis/results.json   - every number quoted in the thesis
    thesis/figures/*.png           - the result charts
"""
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

SEED = 42
OUT_JSON = "thesis/analysis/results.json"
FIG_DIR = "thesis/figures"
BASE = ['maf_year', 'vehicle_age', 'mileage_km', 'previous_owners',
        'is_reconditioned', 'power_shutters', 'power_mirrors', 'reg_gap']
DERIVED = ['mileage_per_year', 'log_mileage_km', 'log_vehicle_age']
TARGET = 'price_million'
R = {}
os.makedirs(FIG_DIR, exist_ok=True)


def add_derived(df):
    df = df.copy()
    df['mileage_per_year'] = df['mileage_km'] / (df['vehicle_age'] + 1)
    df['log_mileage_km'] = np.log1p(df['mileage_km'])
    df['log_vehicle_age'] = np.log1p(df['vehicle_age'])
    return df


def metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ape = np.abs((y - p) / y)
    return {
        "mae_lkr": float(mean_absolute_error(y, p) * 1e6),
        "rmse_lkr": float(np.sqrt(mean_squared_error(y, p)) * 1e6),
        "r2": float(r2_score(y, p)),
        "mape_pct": float(ape.mean() * 100),
        "within_2_5_pct": float((ape <= 0.025).mean() * 100),
        "within_5_pct": float((ape <= 0.05).mean() * 100),
        "within_10_pct": float((ape <= 0.10).mean() * 100),
        "max_abs_err_lkr": float(np.abs(y - p).max() * 1e6),
    }


# ----------------------------------------------------------------- data
clean = pd.read_csv("data/processed/alto_clean.csv")
hol = pd.read_csv("data/processed/holistic_augmented.csv")
feats = json.load(open("models/saved/advanced_feature_names.json"))
pipe = joblib.load("models/saved/advanced_best_model.joblib")

hol_d = add_derived(hol)
X, y = hol_d[feats], hol_d[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED)
src_te = hol.loc[X_te.index, 'data_source']

R["dataset"] = {
    "real_rows": int(len(clean)),
    "holistic_rows": int(len(hol)),
    "holistic_sources": hol['data_source'].value_counts().to_dict(),
    "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
    "test_sources": src_te.value_counts().to_dict(),
    "price_mean_real": float(clean[TARGET].mean()),
    "price_std_real": float(clean[TARGET].std()),
}

# ------------------------------------------------- 1. saved pipeline, hold-out
p_te = pipe.predict(X_te)
p_tr = pipe.predict(X_tr)
R["stack_test"] = metrics(y_te, p_te)
R["stack_train"] = metrics(y_tr, p_tr)
R["stack_test_by_source"] = {s: metrics(y_te[src_te == s], p_te[(src_te == s).values])
                             for s in src_te.unique()}

# ------------------------------------------------- 2. base learners inside the stack
pre = pipe.named_steps['preprocessor']
stack = pipe.named_steps['model']
Xte_p = pre.transform(X_te)
names = [n for n, _ in stack.estimators]
R["base_learners"] = {}
R["tuned_params"] = {}
keep = {'rf': ['n_estimators', 'max_depth', 'min_samples_split'],
        'gb': ['n_estimators', 'learning_rate', 'max_depth'],
        'xgb': ['n_estimators', 'learning_rate', 'max_depth'],
        'lgbm': ['n_estimators', 'learning_rate', 'num_leaves']}
for n, est in zip(names, stack.estimators_):
    R["base_learners"][n] = metrics(y_te, est.predict(Xte_p))
    R["tuned_params"][n] = {k: est.get_params()[k] for k in keep[n]}
R["meta_learner"] = {"coef": dict(zip(names, map(float, stack.final_estimator_.coef_))),
                     "intercept": float(stack.final_estimator_.intercept_)}

# ------------------------------------------------- 3. simple reference baselines
R["baselines"] = {}
R["baselines"]["global_mean"] = metrics(y_te, np.full(len(y_te), y_tr.mean()))
ym = pd.concat([X_tr['maf_year'], y_tr], axis=1).groupby('maf_year')[TARGET].mean()
R["baselines"]["year_mean"] = metrics(y_te, X_te['maf_year'].map(ym).fillna(y_tr.mean()))
lin = Pipeline([('pre', clone(pre)), ('m', LinearRegression())]).fit(X_tr, y_tr)
R["baselines"]["linear_regression"] = metrics(y_te, lin.predict(X_te))

# ------------------------------------------------- 4. permutation importance
pi = permutation_importance(pipe, X_te, y_te, n_repeats=20, random_state=SEED,
                            scoring='neg_mean_absolute_error')
R["permutation_importance_lkr"] = {
    f: {"mean": float(m * 1e6), "std": float(s * 1e6)}
    for f, m, s in sorted(zip(feats, pi.importances_mean, pi.importances_std),
                          key=lambda t: -t[1])}

# ------------------------------------------------- 5. sensitivity (what-if) analysis
repair = json.load(open("data/repair_costs.json", encoding="utf-8"))
ref = dict(maf_year=2015, vehicle_age=11, mileage_km=100000, previous_owners=2,
           is_reconditioned=0, power_shutters=1, power_mirrors=0, reg_gap=0,
           fault_class='healthy', confidence=1.0, body_score=100)


def predict_rows(rows):
    return pipe.predict(add_derived(pd.DataFrame(rows))[feats]) * 1e6


base_pred = float(predict_rows([ref])[0])
R["sensitivity"] = {"reference_vehicle": ref, "reference_price_lkr": base_pred}

scores = list(range(40, 101, 5))
bs = predict_rows([{**ref, 'body_score': s} for s in scores])


def rule_body(s):
    cat = "none" if s >= 80 else "minor" if s >= 60 else "moderate" if s >= 40 else "severe"
    c = repair["body_damage_categories"][cat]
    return (c["min"] + c["max"]) / 2


R["sensitivity"]["body_score"] = [
    {"score": s, "pred_lkr": float(p), "model_drop_lkr": float(base_pred - p),
     "rule_deduction_lkr": float(rule_body(s))} for s, p in zip(scores, bs)]

faults = list(repair["engine_faults"].keys())
R["sensitivity"]["fault_class"] = []
for f in faults:
    conf = 1.0 if f == "healthy" else 0.9
    p = float(predict_rows([{**ref, 'fault_class': f, 'confidence': conf}])[0])
    c = repair["engine_faults"][f]
    R["sensitivity"]["fault_class"].append(
        {"fault": f, "confidence": conf, "pred_lkr": p, "model_drop_lkr": base_pred - p,
         "rule_deduction_lkr": (c["min"] + c["max"]) / 2 * conf})

miles = list(range(40000, 200001, 20000))
mp = predict_rows([{**ref, 'mileage_km': m} for m in miles])
R["sensitivity"]["mileage"] = [{"mileage_km": m, "pred_lkr": float(p)} for m, p in zip(miles, mp)]
years = list(range(2012, 2018))
yp = predict_rows([{**ref, 'maf_year': yr, 'vehicle_age': 2026 - yr} for yr in years])
R["sensitivity"]["year"] = [{"maf_year": yr, "pred_lkr": float(p)} for yr, p in zip(years, yp)]


# ------------------------------------------------- 6. verdict behaviour on the hold-out
# A listing priced exactly at its recorded price should come back FAIR_PRICE.
# A listing inflated or discounted by a known margin should be flagged.
def verdict(listed, fair):
    if listed > fair * 1.05:
        return "OVERPRICED"
    if listed < fair * 0.95:
        return "GOOD_DEAL"
    return "FAIR_PRICE"


R["verdict_simulation"] = {}
for label, factor in [("at_market", 1.00), ("plus_10pct", 1.10), ("plus_15pct", 1.15),
                      ("minus_10pct", 0.90), ("minus_15pct", 0.85)]:
    v = pd.Series([verdict(a * factor, f) for a, f in zip(y_te.values, p_te)])
    R["verdict_simulation"][label] = (v.value_counts(normalize=True) * 100).round(2).to_dict()

# ------------------------------------------------- 7. real listings only (no synthetic rows)
clean_d = add_derived(clean)
real_feats = BASE + DERIVED
num_pre = ColumnTransformer([('num', StandardScaler(), real_feats)])


def fresh_stack():
    return StackingRegressor(estimators=[(n, clone(e)) for n, e in zip(names, stack.estimators_)],
                             final_estimator=Ridge(), cv=5, n_jobs=-1)


candidates = {"Linear regression": LinearRegression()}
label = {'rf': 'Random Forest', 'gb': 'Gradient Boosting', 'xgb': 'XGBoost', 'lgbm': 'LightGBM'}
for n, e in zip(names, stack.estimators_):
    candidates[label[n]] = clone(e)
candidates["Stacking ensemble"] = fresh_stack()

R["real_only_cv"] = {}
Xr, yr_ = clean_d[real_feats], clean_d[TARGET]
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for name, est in candidates.items():
    oof = np.zeros(len(Xr))
    for tr, te in kf.split(Xr):
        m = Pipeline([('pre', clone(num_pre)), ('m', clone(est))]).fit(Xr.iloc[tr], yr_.iloc[tr])
        oof[te] = m.predict(Xr.iloc[te])
    R["real_only_cv"][name] = metrics(yr_, oof)
oof = np.zeros(len(Xr))
for tr, te in kf.split(Xr):
    by_year = yr_.iloc[tr].groupby(Xr.iloc[tr]['maf_year']).mean()
    oof[te] = Xr.iloc[te]['maf_year'].map(by_year).fillna(yr_.iloc[tr].mean()).values
R["real_only_cv"]["Year-mean baseline"] = metrics(yr_, oof)

# ------------------------------------------------- 8. random split versus grouped split
# Rebuild the holistic set with the generator's own logic so that each synthetic row
# carries the index of the real listing it was derived from.
from preprocessing.augment_holistic import generate_holistic_row  # noqa: E402

rng = np.random.default_rng(SEED)
rows = []
for idx, row in clean.iterrows():
    d = row.copy().to_dict()
    d.update(fault_class='healthy', confidence=1.0, body_score=100, src=idx)
    rows.append(d)
for idx in rng.choice(clean.index, size=2000 - len(rows), replace=True):
    d = generate_holistic_row(clean.loc[idx], rng)
    d['src'] = idx
    rows.append(d)
regen = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
same = bool(np.allclose(regen[TARGET].values, hol[TARGET].values)
            and (regen['fault_class'].values == hol['fault_class'].values).all())
R["split_comparison"] = {"regenerated_matches_saved_file": same}

regen_d = add_derived(regen)
Xg, yg, groups = regen_d[feats], regen_d[TARGET], regen_d['src']
full_pre = ColumnTransformer([('num', StandardScaler(), [c for c in feats if c != 'fault_class']),
                              ('cat', OneHotEncoder(handle_unknown='ignore'), ['fault_class'])])


def cv_eval(splitter, **kw):
    oof = np.zeros(len(Xg))
    fold_mae = []
    for tr, te in splitter.split(Xg, yg, **kw):
        m = Pipeline([('pre', clone(full_pre)), ('m', fresh_stack())]).fit(Xg.iloc[tr], yg.iloc[tr])
        oof[te] = m.predict(Xg.iloc[te])
        fold_mae.append(float(mean_absolute_error(yg.iloc[te], oof[te]) * 1e6))
    out = metrics(yg, oof)
    out["fold_mae_lkr"] = fold_mae
    out["fold_mae_std_lkr"] = float(np.std(fold_mae))
    return out


R["split_comparison"]["random_5fold"] = cv_eval(KFold(5, shuffle=True, random_state=SEED))
R["split_comparison"]["grouped_5fold"] = cv_eval(GroupKFold(5), groups=groups)
# share of hold-out rows whose source listing also contributed rows to the training part
tr_idx, te_idx = train_test_split(np.arange(len(regen)), test_size=0.20, random_state=SEED)
R["split_comparison"]["holdout_rows_with_source_in_train_pct"] = float(
    regen.loc[te_idx, 'src'].isin(set(regen.loc[tr_idx, 'src'])).mean() * 100)
R["split_comparison"]["rows_per_source_mean"] = float(regen.groupby('src').size().mean())

# ------------------------------------------------- 9. functional tests and timing
from inference.valuate import load_all, valuate  # noqa: E402

assets = load_all()
veh = {"maf_year": 2015, "reg_year": 2015, "mileage_km": 85000, "previous_owners": 2,
       "is_reconditioned": 0, "power_shutters": 1, "power_mirrors": 0}
cases = [
    ("TC-01", "Healthy car listed near model value", dict(listed_price_million=3.95)),
    ("TC-02", "Same car listed well above value", dict(listed_price_million=4.60)),
    ("TC-03", "Same car listed well below value", dict(listed_price_million=3.30)),
    ("TC-04", "Knocking fault, body 82, listed 3.89 M",
     dict(listed_price_million=3.89, fault_class="knocking", confidence=0.87, body_score=82)),
    ("TC-05", "Altered VIN", dict(listed_price_million=3.80, vin_status="altered")),
    ("TC-06", "VIN needs review", dict(listed_price_million=3.95, vin_status="need review")),
    ("TC-07", "Severe body damage (score 45)", dict(listed_price_million=3.95, body_score=45)),
    ("TC-08", "Unknown fault label", dict(listed_price_million=3.95, fault_class="overheating", confidence=0.8)),
]
R["functional_cases"] = []
for cid, desc, kw in cases:
    r = valuate(veh, preloaded_assets=assets, **kw)
    R["functional_cases"].append({
        "id": cid, "description": desc, "inputs": kw, "verdict": r["verdict"],
        "fair_value_lkr": r.get("fair_value_lkr"), "recommended_offer_lkr": r.get("recommended_offer_lkr"),
        "negotiation_min_lkr": r.get("negotiation_min_lkr"), "negotiation_max_lkr": r.get("negotiation_max_lkr"),
        "body_damage_category": r.get("body_damage_category"), "engine_deduction_lkr": r.get("engine_deduction_lkr"),
        "body_deduction_lkr": r.get("body_deduction_lkr"), "vin_warning": r.get("vin_warning"),
        "explanation": r.get("explanation"), "verdict_message": r.get("verdict_message")})

lat = []
for _ in range(200):
    t = time.perf_counter()
    valuate(veh, listed_price_million=3.89, fault_class="knocking", confidence=0.87,
            body_score=82, preloaded_assets=assets)
    lat.append((time.perf_counter() - t) * 1000)
R["latency_ms_inprocess"] = {"n": 200, "mean": float(np.mean(lat)), "median": float(np.median(lat)),
                             "p95": float(np.percentile(lat, 95)), "max": float(np.max(lat))}

import api.app as api_app  # noqa: E402

client = api_app.app.test_client()
good = {"Authorization": f"Bearer {api_app.API_TOKEN}"}
body = {**veh, "listed_price_million": 3.89, "fault_class": "knocking", "confidence": 0.87,
        "body_score": 82, "vin_status": "original"}
api_cases = [
    ("API-01", "GET /api/v1/health", lambda: client.get("/api/v1/health"), 200),
    ("API-02", "POST /valuate without Authorization header", lambda: client.post("/api/v1/valuate", json=body), 401),
    ("API-03", "POST /valuate with wrong token",
     lambda: client.post("/api/v1/valuate", json=body, headers={"Authorization": "Bearer wrong"}), 401),
    ("API-04", "POST /valuate with valid token and full body",
     lambda: client.post("/api/v1/valuate", json=body, headers=good), 200),
    ("API-05", "POST /valuate with mileage_km missing",
     lambda: client.post("/api/v1/valuate", json={k: v for k, v in body.items() if k != "mileage_km"}, headers=good), 400),
    ("API-06", "POST /valuate with listed_price_million missing",
     lambda: client.post("/api/v1/valuate", json={k: v for k, v in body.items() if k != "listed_price_million"}, headers=good), 400),
    ("API-07", "POST /valuate with non-JSON body",
     lambda: client.post("/api/v1/valuate", data="x", headers=good), 400),
    ("API-08", "POST /valuate with non-numeric price",
     lambda: client.post("/api/v1/valuate", json={**body, "listed_price_million": "abc"}, headers=good), 400),
    ("API-09", "POST /valuate with altered VIN",
     lambda: client.post("/api/v1/valuate", json={**body, "vin_status": "altered"}, headers=good), 200),
    ("API-10", "GET unknown route", lambda: client.get("/api/v1/nope"), 404),
    ("API-11", "POST /valuate with only the seven vehicle fields and price (defaults used)",
     lambda: client.post("/api/v1/valuate", json={**veh, "listed_price_million": 3.89}, headers=good), 200),
]
R["api_cases"] = []
for cid, desc, fn, exp in api_cases:
    resp = fn()
    js = resp.get_json(silent=True) or {}
    R["api_cases"].append({"id": cid, "description": desc, "expected_status": exp,
                           "actual_status": resp.status_code, "passed": resp.status_code == exp,
                           "verdict": js.get("verdict"), "error": js.get("error")})
lat = []
for _ in range(200):
    t = time.perf_counter()
    client.post("/api/v1/valuate", json=body, headers=good)
    lat.append((time.perf_counter() - t) * 1000)
R["latency_ms_api"] = {"n": 200, "mean": float(np.mean(lat)), "median": float(np.median(lat)),
                       "p95": float(np.percentile(lat, 95)), "max": float(np.max(lat))}
R["model_file_mb"] = os.path.getsize("models/saved/advanced_best_model.joblib") / 1e6

json.dump(R, open(OUT_JSON, "w"), indent=2, default=str)
print("results written to", OUT_JSON)

# ----------------------------------------------------------------- figures
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
                     "axes.axisbelow": True})
INK, MID, LIGHT, ACC = "#1f3b57", "#5b7f9e", "#b9c9d8", "#b5532a"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight")
    plt.close(fig)


# price by year of manufacture (real listings)
fig, ax = plt.subplots(figsize=(6.2, 3.6))
yrs = sorted(clean.maf_year.unique())
ax.boxplot([clean.loc[clean.maf_year == v, TARGET] for v in yrs],
           patch_artist=True, boxprops=dict(facecolor=LIGHT, edgecolor=INK),
           medianprops=dict(color=ACC, linewidth=1.6), whiskerprops=dict(color=INK),
           capprops=dict(color=INK), flierprops=dict(marker='o', markersize=3, markeredgecolor=MID))
ax.set_xticks(range(1, len(yrs) + 1))
ax.set_xticklabels([str(v) for v in yrs])
for i, v in enumerate(yrs, 1):
    ax.text(i, 2.86, f"n={int((clean.maf_year == v).sum())}", ha="center", fontsize=9, color="#444444")
ax.set_ylim(2.8, 4.6)
ax.set_xlabel("Year of manufacture")
ax.set_ylabel("Listed price (LKR million)")
ax.grid(axis="x", visible=False)
save(fig, "fig_price_by_year.png")

# mileage against price (real listings)
fig, ax = plt.subplots(figsize=(6.2, 3.6))
sc = ax.scatter(clean.mileage_km / 1000, clean[TARGET], c=clean.maf_year, cmap="viridis_r", s=14, alpha=0.85)
cb = fig.colorbar(sc, ax=ax)
cb.set_label("Year of manufacture")
ax.set_xlabel("Mileage (thousand km)")
ax.set_ylabel("Listed price (LKR million)")
save(fig, "fig_mileage_vs_price.png")

# model comparison on the hold-out
order = [("Global mean", R["baselines"]["global_mean"]), ("Year mean", R["baselines"]["year_mean"]),
         ("Linear regression", R["baselines"]["linear_regression"]),
         ("LightGBM", R["base_learners"]["lgbm"]), ("Gradient Boosting", R["base_learners"]["gb"]),
         ("Random Forest", R["base_learners"]["rf"]), ("XGBoost", R["base_learners"]["xgb"]),
         ("Stacking ensemble", R["stack_test"])]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
vals = [m["mae_lkr"] / 1000 for _, m in order]
bars = ax.barh([n for n, _ in order], vals, color=[LIGHT] * 3 + [MID] * 4 + [ACC], height=0.62)
for b, v in zip(bars, vals):
    ax.text(v + 2, b.get_y() + b.get_height() / 2, f"{v:,.0f}", va="center", fontsize=9)
ax.set_xlabel("Mean absolute error on hold-out set (LKR thousand)")
ax.set_xlim(0, max(vals) * 1.12)
ax.grid(axis="y", visible=False)
save(fig, "fig_model_comparison.png")

# predicted against actual
fig, ax = plt.subplots(figsize=(4.8, 4.6))
ax.scatter(y_te, p_te, s=12, color=MID, alpha=0.7, edgecolor="none")
lo, hi = 2.8, 4.6
ax.plot([lo, hi], [lo, hi], color=INK, linewidth=1.1)
ax.plot([lo, hi], [lo * 1.05, hi * 1.05], color=ACC, linewidth=0.9, linestyle="--")
ax.plot([lo, hi], [lo * 0.95, hi * 0.95], color=ACC, linewidth=0.9, linestyle="--")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel("Recorded price (LKR million)")
ax.set_ylabel("Predicted price (LKR million)")
save(fig, "fig_pred_vs_actual.png")

# residual distribution
fig, ax = plt.subplots(figsize=(6.2, 3.4))
res = (p_te - y_te.values) * 1000
ax.hist(res, bins=30, color=MID, edgecolor="white")
ax.axvline(0, color=INK, linewidth=1.1)
ax.set_xlabel("Prediction error: predicted minus recorded (LKR thousand)")
ax.set_ylabel("Hold-out records")
ax.grid(axis="x", visible=False)
save(fig, "fig_residuals.png")
R["residuals"] = {"mean_lkr": float(res.mean() * 1000), "std_lkr": float(res.std() * 1000),
                  "p05_lkr": float(np.percentile(res, 5) * 1000), "p95_lkr": float(np.percentile(res, 95) * 1000)}

# permutation importance
fig, ax = plt.subplots(figsize=(6.2, 4.0))
items = list(R["permutation_importance_lkr"].items())[::-1]
ax.barh([k for k, _ in items], [v["mean"] / 1000 for _, v in items],
        xerr=[v["std"] / 1000 for _, v in items], color=MID, ecolor=INK, height=0.62)
ax.set_xlabel("Increase in hold-out MAE when the feature is shuffled (LKR thousand)")
ax.grid(axis="y", visible=False)
save(fig, "fig_permutation_importance.png")

# sensitivity to body score
fig, ax = plt.subplots(figsize=(6.2, 3.5))
b = R["sensitivity"]["body_score"]
ax.plot([d["score"] for d in b], [d["model_drop_lkr"] / 1000 for d in b], marker="o", color=INK,
        label="Drop predicted by the model")
ax.step([d["score"] for d in b], [d["rule_deduction_lkr"] / 1000 for d in b], where="mid", color=ACC,
        linestyle="--", label="Deduction used when generating the data")
ax.set_xlabel("Body condition score")
ax.set_ylabel("Reduction in value (LKR thousand)")
ax.legend(frameon=False, fontsize=9)
save(fig, "fig_sensitivity_body.png")

# sensitivity to engine fault
fig, ax = plt.subplots(figsize=(6.2, 3.5))
f = R["sensitivity"]["fault_class"]
xs = np.arange(len(f))
w = 0.38
ax.bar(xs - w / 2, [d["model_drop_lkr"] / 1000 for d in f], w, color=INK, label="Drop predicted by the model")
ax.bar(xs + w / 2, [d["rule_deduction_lkr"] / 1000 for d in f], w, color=ACC,
       label="Deduction used when generating the data")
ax.set_xticks(xs)
ax.set_xticklabels([d["fault"].replace("_", "\n") for d in f], fontsize=9)
ax.set_ylabel("Reduction in value (LKR thousand)")
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="x", visible=False)
save(fig, "fig_sensitivity_fault.png")

# sensitivity to mileage and year
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.1))
m = R["sensitivity"]["mileage"]
axs[0].plot([d["mileage_km"] / 1000 for d in m], [d["pred_lkr"] / 1e6 for d in m], marker="o", color=INK)
axs[0].set_xlabel("Mileage (thousand km)")
axs[0].set_ylabel("Predicted value (LKR million)")
yv = R["sensitivity"]["year"]
axs[1].plot([d["maf_year"] for d in yv], [d["pred_lkr"] / 1e6 for d in yv], marker="o", color=INK)
axs[1].set_xlabel("Year of manufacture")
save(fig, "fig_sensitivity_mileage_year.png")

# validation schemes side by side
fig, ax = plt.subplots(figsize=(6.2, 3.3))
lab = ["Hold-out split\n(as trained)", "Random 5-fold\n(2,000 rows)", "Grouped 5-fold\n(2,000 rows)",
       "5-fold on real\nlistings only (338)"]
v = [R["stack_test"]["mae_lkr"], R["split_comparison"]["random_5fold"]["mae_lkr"],
     R["split_comparison"]["grouped_5fold"]["mae_lkr"], R["real_only_cv"]["Stacking ensemble"]["mae_lkr"]]
bars = ax.bar(lab, [x / 1000 for x in v], color=[MID, MID, ACC, ACC], width=0.55)
for bb, x in zip(bars, v):
    ax.text(bb.get_x() + bb.get_width() / 2, x / 1000 + 2, f"{x / 1000:,.0f}", ha="center", fontsize=9)
ax.set_ylabel("MAE of the stacking ensemble (LKR thousand)")
ax.grid(axis="x", visible=False)
ax.set_ylim(0, max(v) / 1000 * 1.15)
save(fig, "fig_validation_schemes.png")

json.dump(R, open(OUT_JSON, "w"), indent=2, default=str)
print("figures written to", FIG_DIR)

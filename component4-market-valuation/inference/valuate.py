"""
Valuation engine.

  fair value = ML market price             (learned from real Suzuki Alto listings)
             - engine repair cost           (component 3 health score x surveyed repair cost)
             - body repair cost             (component 2 body score x surveyed repair cost)
             - VIN risk discount            (only when the VIN needs review)

The parts are kept separate on purpose: listings do not record engine or body
condition, so the market model cannot learn it. Condition is priced from the
repair cost survey (data/repair_costs.json) and every deduction is shown to the
user line by line, together with the factors that moved the market price.
"""
import json
import pathlib
import sys
from datetime import date

import joblib
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
from models.market_model import CURRENT_YEAR_BASE, add_features  # noqa: E402  (also needed to unpickle the model)

MODEL_PATH = "models/saved/market_model.joblib"
META_PATH = "models/saved/market_model_meta.json"
REPAIR_COSTS_PATH = "data/repair_costs.json"
MIN_VALUE_LKR = 500_000
STALE_MARKET_MONTHS = 3     # warn when the newest listings are older than this
MIN_FACTOR_LKR = 5_000      # smaller price effects are not worth showing
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Component 3's Mechanical Health Score formula, used only when the app sends a
# fault class without the score (older clients). Mirrors component3 inference/predict.py.
MHS_BASE_SCORES = {"healthy": 100, "knocking": 55, "misfiring": 50, "rotational_imbalance": 60,
                   "tappet": 65, "battery_fault": 70}
MHS_PENALTY_WEIGHT = 30


def load_all():
    """Load the market model, its metadata and the repair cost survey."""
    try:
        model = joblib.load(MODEL_PATH)
        with open(META_PATH, "r") as f:
            meta = json.load(f)
        with open(REPAIR_COSTS_PATH, "r") as f:
            repair_costs = json.load(f)
        print(f"Loaded market model ({meta['training_rows']} real listings) and repair costs")
        return model, meta, meta["features"], repair_costs
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Required file missing for valuation engine: {e.filename}. "
            "Run preprocessing/clean_real_data.py and models/train_market_model.py first."
        )


def months_since_base(day):
    """Months since Dec 2025, the same scale as listed_months in the training data."""
    return (day.year - CURRENT_YEAR_BASE) * 12 + day.month


def month_label(month):
    year = CURRENT_YEAR_BASE + (month - 1) // 12
    return f"{MONTH_NAMES[(month - 1) % 12]} {year}"


# --------------------------------------------------------------------------- scope

def check_scope(vehicle_input, meta, make=None, model_name=None, today=None):
    """Return a list of reasons the car is outside what the model knows (empty = in scope)."""
    today = today or date.today()
    scope = meta["scope"]
    reasons = []

    if make and str(make).strip().lower() != scope["make"].lower():
        reasons.append(f"The price model only knows {scope['make']} {scope['model']}, not {make}.")
    elif model_name and str(model_name).strip().lower() != scope["model"].lower():
        reasons.append(f"The price model only knows {scope['make']} {scope['model']}, not {make or ''} {model_name}.".replace("  ", " "))

    year_min, year_max = meta["year_range"]
    if not year_min <= vehicle_input["maf_year"] <= year_max:
        reasons.append(f"Manufacture year {vehicle_input['maf_year']} is outside the {year_min}-{year_max} listings the model was trained on.")

    km_min, km_max = scope["mileage_km"]
    if not km_min <= vehicle_input["mileage_km"] <= km_max:
        reasons.append(f"Mileage {vehicle_input['mileage_km']:,} km is outside the {km_min:,}-{km_max:,} km range in the listings.")

    own_min, own_max = scope["previous_owners"]
    if not own_min <= vehicle_input["previous_owners"] <= own_max:
        reasons.append(f"{vehicle_input['previous_owners']} previous owners is outside the {own_min}-{own_max} range in the listings.")

    return reasons


def validate_input(vehicle_input, today=None):
    """Reject impossible inputs (raised as ValueError -> HTTP 400)."""
    today = today or date.today()
    if vehicle_input["reg_year"] < vehicle_input["maf_year"]:
        raise ValueError("reg_year cannot be earlier than maf_year")
    if vehicle_input["reg_year"] > today.year or vehicle_input["maf_year"] > today.year:
        raise ValueError("maf_year and reg_year cannot be in the future")
    if vehicle_input["mileage_km"] < 0 or vehicle_input["previous_owners"] < 0:
        raise ValueError("mileage_km and previous_owners cannot be negative")


# --------------------------------------------------------------------------- market price

def _feature_row(vehicle, meta, today):
    row = pd.DataFrame([{
        "maf_year": vehicle["maf_year"],
        "reg_year": vehicle["reg_year"],
        "mileage_km": vehicle["mileage_km"],
        "previous_owners": vehicle["previous_owners"],
        "is_reconditioned": vehicle["is_reconditioned"],
        "power_shutters": vehicle["power_shutters"],
        "power_mirrors": vehicle["power_mirrors"],
        "listed_months": meta["market_month"],          # market level: newest listings seen
        "age_ref_months": months_since_base(today),     # vehicle age: today
    }])
    return add_features(row)


def predict_market_price(model, meta, vehicle_input, today=None):
    """Market price of the vehicle in good condition, with the 80% range of real listings."""
    today = today or date.today()
    price = float(model.predict(_feature_row(vehicle_input, meta, today))[0])
    interval = meta["price_interval"]
    return {
        "raw_lkr": price,
        "market_value_lkr": int(round(price, -3)),
        "market_low_lkr": int(round(price * (1 + interval["low_ratio"]), -3)),
        "market_high_lkr": int(round(price * (1 + interval["high_ratio"]), -3)),
    }


def explain_market_price(model, meta, vehicle_input, base_price, today=None):
    """
    How much each factor moves this car's price compared with a typical listing.
    For each factor the car's value is swapped for the typical value and the model
    is asked again; the difference is that factor's effect on the price.
    """
    today = today or date.today()
    ref = meta["reference_vehicle"]
    v = vehicle_input
    reg_gap = max(0, v["reg_year"] - v["maf_year"])
    groups = [
        ("year", f"{v['maf_year']} model year", f"{ref['maf_year']}",
         {"maf_year": ref["maf_year"], "reg_year": ref["maf_year"] + reg_gap}),
        ("mileage", f"{v['mileage_km']:,} km", f"{ref['mileage_km']:,} km",
         {"mileage_km": ref["mileage_km"]}),
        ("owners", f"{v['previous_owners']} previous owner{'s' if v['previous_owners'] != 1 else ''}",
         f"{ref['previous_owners']}", {"previous_owners": ref["previous_owners"]}),
        ("registration", f"registered {reg_gap} year{'s' if reg_gap != 1 else ''} after manufacture",
         f"{ref['reg_gap']} years", {"reg_year": v["maf_year"] + ref["reg_gap"]}),
        ("features", "power shutters " + ("yes" if v["power_shutters"] else "no") + ", power mirrors "
         + ("yes" if v["power_mirrors"] else "no"), "shutters yes, mirrors no",
         {"power_shutters": ref["power_shutters"], "power_mirrors": ref["power_mirrors"]}),
        ("reconditioned", "reconditioned" if v["is_reconditioned"] else "brand new when registered",
         "brand new", {"is_reconditioned": ref["is_reconditioned"]}),
    ]
    factors = []
    for key, value_text, typical_text, swap in groups:
        if all(v[k] == val for k, val in swap.items()):
            continue
        typical_price = float(model.predict(_feature_row({**v, **swap}, meta, today))[0])
        impact = int(round(base_price - typical_price, -3))
        if abs(impact) >= MIN_FACTOR_LKR:
            sign = "+" if impact > 0 else "-"
            factors.append({
                "factor": key,
                "impact_lkr": impact,
                "text": f"{sign}LKR {abs(impact):,} for {value_text} (typical: {typical_text})",
            })
    return sorted(factors, key=lambda f: -abs(f["impact_lkr"]))


# --------------------------------------------------------------------------- condition

def estimate_mhs(fault_class, confidence):
    if fault_class == "healthy":
        return int(round(confidence * 100))
    base = MHS_BASE_SCORES.get(fault_class)
    if base is None:
        return 0
    return max(0, min(100, int(round(base - MHS_PENALTY_WEIGHT * confidence))))


def get_engine_deduction(fault_class, confidence, repair_costs, mhs_score=None):
    """
    Engine repair cost from component 3's health score colour:
      green -> nothing, amber -> half the typical engine repair, red -> the full typical repair.
    The fault name is only shown as a possible cause.
    """
    health = repair_costs["engine_health"]
    if mhs_score is None:
        mhs_score = estimate_mhs(fault_class, confidence)
    mhs_score = float(mhs_score)
    color, band = next((c, b) for c, b in sorted(health["bands"].items(), key=lambda kv: -kv[1]["min_mhs"])
                       if mhs_score >= b["min_mhs"])
    share = band["share"]
    repair_min = int(health["typical_min"] * share) if share >= 1 else 0
    repair_max = int(health["typical_max"] * share)

    description = band["description"]
    if fault_class and fault_class != "healthy" and color != "green":
        description += f" (possible cause: {fault_class.replace('_', ' ')})"

    return {
        "fault_class": fault_class,
        "confidence": confidence,
        "mhs_score": mhs_score,
        "health_color": color,
        "repair_min_lkr": repair_min,
        "repair_max_lkr": repair_max,
        "deduction_lkr": int(round(health["expected"] * share, -3)),
        "description": description,
    }


def body_category(body_score, repair_costs):
    categories = repair_costs.get("body_damage_categories", {})
    bands = sorted(
        ((name, data) for name, data in categories.items() if isinstance(data, dict)),
        key=lambda item: -item[1]["min_score"],
    )
    for name, data in bands:
        if body_score >= data["min_score"]:
            return name, data
    return bands[-1]


def get_body_deduction(body_score, repair_costs):
    """Body repair cost for the damage category that component 2's score falls in."""
    category, cat_data = body_category(body_score, repair_costs)
    repair_min, repair_max = cat_data["min"], cat_data["max"]
    return {
        "body_score": body_score,
        "damage_category": category,
        "repair_min_lkr": repair_min,
        "repair_max_lkr": repair_max,
        "deduction_lkr": int((repair_min + repair_max) / 2),
        "description": cat_data["description"],
    }


# --------------------------------------------------------------------------- output

def generate_explanation(verdict, listed_lkr, fair_value_lkr, engine_result, body_result, vin_status):
    """Plain English explanation for negotiation (at most 3 sentences)."""
    diff_abs = abs(listed_lkr - fair_value_lkr)
    if verdict == "GOOD_DEAL":
        sentences = [f"This is a good deal: the listed price is LKR {diff_abs:,} below the estimated fair value of LKR {fair_value_lkr:,}"]
    elif verdict == "OVERPRICED":
        sentences = [f"This vehicle is overpriced by LKR {diff_abs:,} compared with its estimated fair value of LKR {fair_value_lkr:,}"]
    else:
        sentences = [f"The listed price is within the normal market range for this vehicle (fair value LKR {fair_value_lkr:,})"]

    if engine_result["health_color"] != "green":
        sentences.append(
            f"The engine check flagged a {'likely' if engine_result['health_color'] == 'red' else 'possible'} problem; "
            f"engine repairs typically cost up to LKR {engine_result['repair_max_lkr']:,}"
        )
    if body_result["damage_category"] != "none":
        sentences.append(
            f"Body condition is {body_result['damage_category']} damage, typically LKR "
            f"{body_result['repair_min_lkr']:,}-{body_result['repair_max_lkr']:,} to repair"
        )
    if vin_status in ("need review", "needs review"):
        sentences.append("The VIN could not be confirmed as original, so the fair value includes a risk discount")
    elif vin_status != "original":
        sentences.append("The VIN was not checked, so confirm it at the RMV before paying")
    return ". ".join(sentences[:3]) + "."


def valuate(vehicle_input, listed_price_million, fault_class="healthy", confidence=1.0, body_score=100,
            vin_status="original", preloaded_assets=None, mhs_score=None, make=None, model_name=None, today=None):
    """Main valuation logic."""
    today = today or date.today()
    listed_lkr = int(listed_price_million * 1_000_000)
    vin_status = (vin_status or "unknown").strip().lower()
    validate_input(vehicle_input, today)

    # Step 1 - VIN check: an altered VIN stops the valuation until the VIN is checked
    if vin_status == "altered":
        return {
            "status": "warning",
            "verdict": "CHECK_VIN",
            "verdict_message": "There is an issue with this vehicle's VIN number. Please check this vehicle further before buying.",
            "explanation": ("The VIN plate may have been altered, so no fair price is shown yet. "
                            "Compare the VIN with the registration book, or ask the RMV or a trusted mechanic to verify it."),
            "listed_price_lkr": listed_lkr,
            "vin_status": vin_status,
        }
    vin_needs_review = vin_status in ("need review", "needs review")
    if vin_needs_review:
        vin_warning = "VIN authenticity could not be confirmed. A risk discount is applied; verify the VIN at the RMV before buying."
    elif vin_status != "original":
        vin_warning = "The VIN was not checked. Confirm it matches the registration book before paying."
    else:
        vin_warning = None

    # Step 2 - Load assets
    model, meta, _, repair_costs = preloaded_assets if preloaded_assets else load_all()

    # Step 3 - Refuse cars the model has never seen
    scope_reasons = check_scope(vehicle_input, meta, make, model_name, today)
    if scope_reasons:
        return {
            "status": "out_of_scope",
            "verdict": None,
            "message": "This car can't be valued reliably. " + " ".join(scope_reasons),
            "scope_reasons": scope_reasons,
            "listed_price_lkr": listed_lkr,
        }

    # Step 4 - ML market price (good condition) and what drives it
    market = predict_market_price(model, meta, vehicle_input, today)
    price_factors = explain_market_price(model, meta, vehicle_input, market["raw_lkr"], today)

    # Step 5 - Condition costs from the repair survey
    engine_result = get_engine_deduction(fault_class, confidence, repair_costs, mhs_score)
    body_result = get_body_deduction(body_score, repair_costs)
    condition_lkr = int(round(engine_result["deduction_lkr"] + body_result["deduction_lkr"], -3))
    condition_max_lkr = engine_result["repair_max_lkr"] + body_result["repair_max_lkr"]
    condition_min_lkr = engine_result["repair_min_lkr"] + body_result["repair_min_lkr"]

    # Step 6 - Fair value and fair range, with the VIN risk discount if needed
    vin_factor = 1 - repair_costs["vin_policy"]["needs_review_discount_pct"] / 100 if vin_needs_review else 1.0
    def adjusted(market_lkr, cost_lkr):
        return max(MIN_VALUE_LKR, int(round((market_lkr - cost_lkr) * vin_factor, -3)))
    fair_value_lkr = adjusted(market["market_value_lkr"], condition_lkr)
    fair_low_lkr = adjusted(market["market_low_lkr"], condition_lkr)
    fair_high_lkr = adjusted(market["market_high_lkr"], condition_lkr)
    vin_discount_lkr = int(round((market["market_value_lkr"] - condition_lkr) * (1 - vin_factor), -3))
    price_difference_lkr = listed_lkr - fair_value_lkr

    # Step 7 - Verdict: outside the range where 80% of comparable real listings fall
    if listed_lkr > fair_high_lkr:
        verdict = "OVERPRICED"
        verdict_message = f"This vehicle is listed LKR {price_difference_lkr:,} above its estimated fair value."
    elif listed_lkr < fair_low_lkr and not vin_needs_review:
        verdict = "GOOD_DEAL"
        verdict_message = f"This vehicle is listed LKR {abs(price_difference_lkr):,} below its estimated fair value."
    else:
        verdict = "FAIR_PRICE"
        verdict_message = "This vehicle is listed within the normal market range."
        if vin_needs_review and listed_lkr < fair_low_lkr:
            verdict_message = "The price looks low, but the VIN could not be confirmed. Verify it before treating this as a good deal."

    # Step 8 - Negotiation range
    #   opening offer: low end of the market range with worst-case repair costs
    #   walk-away:     fair value with best-case repair costs, never above the listed price
    negotiation_min_lkr = adjusted(market["market_low_lkr"], condition_max_lkr)
    negotiation_max_lkr = min(adjusted(market["market_value_lkr"], condition_min_lkr), listed_lkr)
    negotiation_min_lkr = min(negotiation_min_lkr, negotiation_max_lkr)
    recommended_offer_lkr = round((negotiation_min_lkr + negotiation_max_lkr) / 2 / 10_000) * 10_000
    recommended_offer_lkr = min(max(recommended_offer_lkr, negotiation_min_lkr), negotiation_max_lkr)
    if verdict == "GOOD_DEAL":
        recommended_offer_lkr = negotiation_min_lkr = negotiation_max_lkr = listed_lkr

    # Step 9 - Is the market data still current?
    data_age_months = months_since_base(today) - meta["market_month"]
    market_data_warning = None
    if data_age_months > STALE_MARKET_MONTHS:
        market_data_warning = (f"Market prices are from {month_label(meta['market_month'])} listings "
                               f"({data_age_months} months old). Retrain with new listings.")

    explanation = generate_explanation(verdict, listed_lkr, fair_value_lkr, engine_result, body_result, vin_status)

    return {
        "status": "success",
        "verdict": verdict,
        "verdict_message": verdict_message,
        "explanation": explanation,

        # Part 1: machine learning market price (good condition) and what drives it
        "base_market_value_lkr": market["market_value_lkr"],
        "base_market_value_million": round(market["market_value_lkr"] / 1_000_000, 3),
        "market_low_lkr": market["market_low_lkr"],
        "market_high_lkr": market["market_high_lkr"],
        "market_as_of": month_label(meta["market_month"]),
        "market_data_warning": market_data_warning,
        "price_factors": price_factors,

        # Part 2: condition adjustment from the repair cost survey
        "engine_fault": fault_class,
        "engine_confidence_percent": f"{int(confidence * 100)}%",
        "engine_mhs_score": engine_result["mhs_score"],
        "engine_health_color": engine_result["health_color"],
        "engine_repair_min_lkr": engine_result["repair_min_lkr"],
        "engine_repair_max_lkr": engine_result["repair_max_lkr"],
        "engine_deduction_lkr": engine_result["deduction_lkr"],
        "engine_description": engine_result["description"],

        "body_score": body_score,
        "body_damage_category": body_result["damage_category"],
        "body_repair_min_lkr": body_result["repair_min_lkr"],
        "body_repair_max_lkr": body_result["repair_max_lkr"],
        "body_deduction_lkr": body_result["deduction_lkr"],

        "condition_deduction_lkr": condition_lkr,

        # Part 3: VIN risk
        "vin_status": vin_status,
        "vin_warning": vin_warning,
        "vin_discount_lkr": vin_discount_lkr,

        "listed_price_lkr": listed_lkr,
        "listed_price_million": listed_price_million,
        "fair_value_lkr": fair_value_lkr,
        "fair_low_lkr": fair_low_lkr,
        "fair_high_lkr": fair_high_lkr,
        "price_difference_lkr": price_difference_lkr,

        "negotiation_min_lkr": negotiation_min_lkr,
        "negotiation_max_lkr": negotiation_max_lkr,
        "recommended_offer_lkr": recommended_offer_lkr,

        "data_source": (
            f"Market price: {meta['model']} trained on {meta['training_rows']} real Suzuki Alto listings "
            f"({meta['year_range'][0]}-{meta['year_range'][1]}), typical error {meta['cv_mape_pct']:.1f}%. "
            "Repair costs: field survey of 5 local sellers and repair centres, Sri Lanka 2026."
        ),
    }


if __name__ == "__main__":
    print("Running valuation engine examples...\n")
    alto = dict(maf_year=2015, reg_year=2015, mileage_km=85000, previous_owners=2,
                is_reconditioned=0, power_shutters=1, power_mirrors=0)
    cases = [
        ("Amber engine, minor body damage", alto, dict(listed_price_million=3.89, fault_class="knocking", confidence=0.87, mhs_score=60, body_score=92)),
        ("Healthy, listed near market", alto, dict(listed_price_million=3.95, mhs_score=95)),
        ("Healthy, listed low", alto, dict(listed_price_million=3.50, body_score=95)),
        ("High mileage, 4 owners, overpriced", {**alto, "mileage_km": 160000, "previous_owners": 4}, dict(listed_price_million=4.2)),
        ("VIN needs review", alto, dict(listed_price_million=3.50, vin_status="needs review")),
        ("Not an Alto", alto, dict(listed_price_million=4.0, make="Suzuki", model_name="Swift")),
        ("Altered VIN", alto, dict(listed_price_million=3.80, vin_status="altered")),
    ]
    assets = load_all()
    for title, vehicle, kwargs in cases:
        r = valuate(vehicle, preloaded_assets=assets, **kwargs)
        print(f"{title}: {r['verdict']} ({r['status']})")
        if r["status"] == "success":
            print(f"  Market value  LKR {r['base_market_value_lkr']:,} (range {r['market_low_lkr']:,}-{r['market_high_lkr']:,}, {r['market_as_of']})")
            for f in r["price_factors"]:
                print(f"    {f['text']}")
            print(f"  Condition     -LKR {r['condition_deduction_lkr']:,} (engine {r['engine_deduction_lkr']:,} [{r['engine_health_color']}], "
                  f"body {r['body_deduction_lkr']:,} [{r['body_damage_category']}]), VIN -LKR {r['vin_discount_lkr']:,}")
            print(f"  Fair value    LKR {r['fair_value_lkr']:,} (range {r['fair_low_lkr']:,}-{r['fair_high_lkr']:,})")
            print(f"  Listed        LKR {r['listed_price_lkr']:,} -> offer LKR {r['recommended_offer_lkr']:,} "
                  f"(negotiate {r['negotiation_min_lkr']:,}-{r['negotiation_max_lkr']:,})")
            print(f"  {r['explanation']}")
        else:
            print(f"  {r.get('message') or r.get('verdict_message')}")
        print()

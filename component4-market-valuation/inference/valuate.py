import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path

# Configuration
MODEL_PATH = "models/saved/price_model.joblib"
SCALER_PATH = "models/saved/price_scaler.joblib"
FEATURE_NAMES_PATH = "models/saved/feature_names.json"
REPAIR_COSTS_PATH = "data/repair_costs.json"
CURRENT_YEAR = 2026

def load_all():
    """
    Load price_model.joblib, price_scaler.joblib, feature_names.json, and repair_costs.json.
    """
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
        
        scaler = joblib.load(SCALER_PATH)
        print(f"Loaded scaler from {SCALER_PATH}")
        
        with open(FEATURE_NAMES_PATH, 'r') as f:
            features = json.load(f)
        print(f"Loaded feature names from {FEATURE_NAMES_PATH}")
        
        with open(REPAIR_COSTS_PATH, 'r') as f:
            repair_costs = json.load(f)
        print(f"Loaded repair costs from {REPAIR_COSTS_PATH}")
        
        return model, scaler, features, repair_costs
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file missing for valuation engine: {e.filename}. Please run training script first.")

def predict_base_price(model, scaler, features, vehicle_input):
    """
    Predict price based on vehicle characteristics.
    """
    maf_year = vehicle_input['maf_year']
    reg_year = vehicle_input['reg_year']
    
    vehicle_age = CURRENT_YEAR - maf_year
    reg_gap = max(0, reg_year - maf_year)
    
    # Construct input data in same order as FEATURES
    data = {
        'maf_year': [maf_year],
        'vehicle_age': [vehicle_age],
        'mileage_km': [vehicle_input['mileage_km']],
        'previous_owners': [vehicle_input['previous_owners']],
        'is_reconditioned': [vehicle_input['is_reconditioned']],
        'power_shutters': [vehicle_input['power_shutters']],
        'power_mirrors': [vehicle_input['power_mirrors']],
        'reg_gap': [reg_gap]
    }
    
    df = pd.DataFrame(data, columns=features)
    
    # Scale and predict
    scaled_data = scaler.transform(df)
    predicted_million = float(model.predict(scaled_data)[0])
    predicted_million = round(predicted_million, 3)
    predicted_lkr = int(predicted_million * 1_000_000)
    
    return predicted_million, predicted_lkr

def get_engine_deduction(fault_class, confidence, repair_costs):
    """
    Look up engine fault in repair_costs and calculate deduction.
    """
    faults = repair_costs.get("engine_faults", {})
    fault_data = faults.get(fault_class, faults.get("healthy"))
    
    repair_min = fault_data["min"]
    repair_max = fault_data["max"]
    midpoint = (repair_min + repair_max) / 2
    deduction = round(midpoint * confidence)
    
    return {
        "fault_class": fault_class,
        "confidence": confidence,
        "repair_min_lkr": repair_min,
        "repair_max_lkr": repair_max,
        "deduction_lkr": deduction,
        "description": fault_data["description"],
        "repair_type": fault_data["repair_type"]
    }

def get_body_deduction(body_score, repair_costs):
    """
    Map body score to damage category and calculate deduction.
    """
    if body_score >= 80:
        category = "none"
    elif body_score >= 60:
        category = "minor"
    elif body_score >= 40:
        category = "moderate"
    else:
        category = "severe"
        
    categories = repair_costs.get("body_damage_categories", {})
    cat_data = categories.get(category, categories.get("none"))
    
    repair_min = cat_data["min"]
    repair_max = cat_data["max"]
    midpoint = (repair_min + repair_max) / 2
    
    return {
        "body_score": body_score,
        "damage_category": category,
        "repair_min_lkr": repair_min,
        "repair_max_lkr": repair_max,
        "deduction_lkr": midpoint,
        "description": cat_data["description"]
    }

def generate_explanation(verdict, listed_lkr, fair_value_lkr, engine_result, body_result, vin_status):
    """
    Generate a plain English string for negotiation.
    """
    diff = listed_lkr - fair_value_lkr
    diff_abs = abs(diff)
    if verdict == "GOOD_DEAL":
        explanation = f"This is a good deal. The listed price is LKR {diff_abs:,} below fair market value. You may proceed at the listed price. "
    else:
        status_str = "overpriced" if diff > 0 else "underpriced"
        explanation = f"This vehicle is {status_str} by LKR {diff_abs:,} relative to its fair market value of LKR {fair_value_lkr:,}. "
    
    if engine_result["fault_class"] != "healthy":
        explanation += f"The {engine_result['fault_class']} engine fault requires repairs estimated between LKR {engine_result['repair_min_lkr']:,}-{engine_result['repair_max_lkr']:,}. "
    
    if body_result["damage_category"] != "none":
        explanation += f"Body condition is rated as {body_result['damage_category']} damage, adding LKR {int(body_result['deduction_lkr']):,} in estimated costs. "
    
    if vin_status != "original":
        explanation += "The VIN status is non-original, which is a major risk factor. "
    
    # Truncate if too long (keep under 3 sentences as requested)
    sentences = [s.strip() for s in explanation.split('. ') if s]
    explanation = ". ".join(sentences[:3]) + "."
    
    return explanation

def valuate(vehicle_input, listed_price_million, fault_class="healthy", confidence=1.0, body_score=100, vin_status="original"):
    """
    Main valuation logic.
    """
    # Step 1 — VIN check
    if vin_status == "altered":
        return {
            "status": "warning",
            "verdict": "DO_NOT_BUY",
            "verdict_message": "This vehicle VIN has been altered. This is a serious legal risk. Do not purchase.",
            "listed_price_lkr": int(listed_price_million * 1_000_000)
        }
        
    # Step 2 — Load all models
    model, scaler, features, repair_costs = load_all()
    
    # Step 3 — Predict base price
    predicted_million, predicted_lkr = predict_base_price(model, scaler, features, vehicle_input)
    
    # Step 4 — Get engine deduction
    engine_result = get_engine_deduction(fault_class, confidence, repair_costs)
    
    # Step 5 — Get body deduction
    body_result = get_body_deduction(body_score, repair_costs)
    
    # Step 6 — Calculate all values
    listed_lkr = int(listed_price_million * 1_000_000)
    fair_value_lkr = int(predicted_lkr - engine_result["deduction_lkr"] - body_result["deduction_lkr"])
    fair_value_lkr = max(500_000, fair_value_lkr)
    
    price_difference_lkr = listed_lkr - fair_value_lkr
    
    negotiation_min_lkr = int(predicted_lkr - engine_result["repair_max_lkr"] - body_result["repair_max_lkr"])
    negotiation_max_lkr = int(predicted_lkr - engine_result["repair_min_lkr"] - body_result["repair_min_lkr"])
    
    negotiation_min_lkr = max(500_000, negotiation_min_lkr)
    negotiation_max_lkr = max(500_000, negotiation_max_lkr)
    
    recommended_offer_lkr = round((negotiation_min_lkr + negotiation_max_lkr) / 2 / 10000) * 10000
    
    # Step 7 — Determine verdict
    if listed_lkr > fair_value_lkr * 1.05:
        verdict = "OVERPRICED"
        verdict_message = f"This vehicle is listed LKR {price_difference_lkr:,} above fair market value."
    elif listed_lkr < fair_value_lkr * 0.95:
        verdict = "GOOD_DEAL"
        verdict_message = f"This vehicle is listed LKR {abs(price_difference_lkr):,} below fair market value."
    else:
        verdict = "FAIR_PRICE"
        verdict_message = "This vehicle is listed at approximately fair market value."
        
    # For good deals, buyer should just pay listed price
    # Recommending higher than listed makes no sense
    if verdict == "GOOD_DEAL":
        recommended_offer_lkr = listed_lkr
        negotiation_min_lkr = listed_lkr
        negotiation_max_lkr = listed_lkr

    # Step 8 — Generate explanation
    explanation = generate_explanation(verdict, listed_lkr, fair_value_lkr, engine_result, body_result, vin_status)
    
    # Step 9 — Return complete dictionary
    return {
        "status": "success",
        "verdict": verdict,
        "verdict_message": verdict_message,
        "explanation": explanation,

        "base_market_value_lkr": predicted_lkr,
        "base_market_value_million": predicted_million,

        "engine_fault": fault_class,
        "engine_confidence_percent": f"{int(confidence * 100)}%",
        "engine_repair_min_lkr": engine_result["repair_min_lkr"],
        "engine_repair_max_lkr": engine_result["repair_max_lkr"],
        "engine_deduction_lkr": engine_result["deduction_lkr"],
        "engine_description": engine_result["description"],

        "body_score": body_score,
        "body_damage_category": body_result["damage_category"],
        "body_repair_min_lkr": body_result["repair_min_lkr"],
        "body_repair_max_lkr": body_result["repair_max_lkr"],
        "body_deduction_lkr": int(body_result["deduction_lkr"]),

        "vin_status": vin_status,

        "listed_price_lkr": listed_lkr,
        "listed_price_million": listed_price_million,
        "fair_value_lkr": fair_value_lkr,
        "price_difference_lkr": price_difference_lkr,

        "negotiation_min_lkr": negotiation_min_lkr,
        "negotiation_max_lkr": negotiation_max_lkr,
        "recommended_offer_lkr": recommended_offer_lkr,

        "data_source": "Repair costs: field survey 5 local sellers Sri Lanka 2026. Price model: 338 real Suzuki Alto listings 2012-2017 riyasewana.lk."
    }

if __name__ == "__main__":
    print("Running valuation engine tests...\n")
    
    # Test 1 — Knocking fault overpriced
    vehicle1 = {"maf_year": 2015, "reg_year": 2015, "mileage_km": 85000,
               "previous_owners": 2, "is_reconditioned": 0,
               "power_shutters": 1, "power_mirrors": 0}
    result1 = valuate(vehicle1, listed_price_million=3.89,
                     fault_class="knocking", confidence=0.87,
                     body_score=82, vin_status="original")
    
    print("Test 1 — Knocking fault overpriced:")
    print(f"Verdict: {result1['verdict']}")
    print(f"Fair Value: LKR {result1['fair_value_lkr']:,}")
    print(f"Recommended Offer: LKR {result1['recommended_offer_lkr']:,}")
    print(f"Explanation: {result1['explanation']}\n")
    
    # Test 2 — Healthy engine good deal
    vehicle2 = {"maf_year": 2015, "reg_year": 2015, "mileage_km": 80000,
               "previous_owners": 2, "is_reconditioned": 0,
               "power_shutters": 1, "power_mirrors": 0}
    result2 = valuate(vehicle2, listed_price_million=3.50,
                     fault_class="healthy", confidence=1.0,
                     body_score=90, vin_status="original")
    
    print("Test 2 — Healthy engine good deal:")
    print(f"Verdict: {result2['verdict']}")
    print(f"Fair Value: LKR {result2['fair_value_lkr']:,}")
    print(f"Recommended Offer: LKR {result2['recommended_offer_lkr']:,}")
    print(f"Explanation: {result2['explanation']}\n")
    
    # Test 3 — Altered VIN
    vehicle3 = {"maf_year": 2015, "reg_year": 2015, "mileage_km": 75000,
               "previous_owners": 1, "is_reconditioned": 0,
               "power_shutters": 1, "power_mirrors": 0}
    result3 = valuate(vehicle3, listed_price_million=3.80,
                     fault_class="healthy", confidence=1.0,
                     body_score=95, vin_status="altered")
    
    print("Test 3 — Altered VIN:")
    print(f"Verdict: {result3['verdict']}")
    if result3['status'] == 'warning':
        print(f"Message: {result3['verdict_message']}")
    else:
        print(f"Fair Value: LKR {result3['fair_value_lkr']:,}")
        print(f"Recommended Offer: LKR {result3['recommended_offer_lkr']:,}")
    print(f"Explanation: {result3.get('explanation', 'N/A')}\n")
    
    print("All valuation tests complete")

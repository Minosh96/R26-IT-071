import pandas as pd
import numpy as np
import os
import json

# Configuration
INPUT_PATH = "data/processed/alto_clean.csv"
OUTPUT_PATH = "data/processed/holistic_augmented.csv"
TARGET_TOTAL_ROWS = 2000 # Increased slightly to help the model learn the new features
RANDOM_SEED = 42

with open("data/repair_costs.json", "r") as f:
    repair_costs = json.load(f)

FAULTS = list(repair_costs["engine_faults"].keys())
# Weighted probability to make most cars healthy
FAULT_PROBS = [0.60] + [0.40 / (len(FAULTS) - 1)] * (len(FAULTS) - 1)

def get_engine_deduction(fault_class, confidence):
    faults = repair_costs.get("engine_faults", {})
    fault_data = faults.get(fault_class, faults.get("healthy"))
    midpoint = (fault_data["min"] + fault_data["max"]) / 2
    return round(midpoint * confidence)

def get_body_deduction(body_score):
    if body_score >= 80:
        category = "none"
    elif body_score >= 60:
        category = "minor"
    elif body_score >= 40:
        category = "moderate"
    else:
        category = "severe"
    cat_data = repair_costs.get("body_damage_categories", {}).get(category, {"min": 0, "max": 0})
    return (cat_data["min"] + cat_data["max"]) / 2

def generate_holistic_row(real_row, rng):
    new_row = real_row.copy().to_dict()
    
    # Standard variation
    mileage_val = real_row['mileage_km']
    mileage_noise = rng.normal(0, 0.05 * mileage_val)
    new_row['mileage_km'] = int(np.clip(round((mileage_val + mileage_noise) / 100) * 100, 5000, 250000))
    
    price_val = real_row['price_million']
    price_noise = rng.normal(0, 0.015 * price_val)
    base_price_million = float(np.clip(round(price_val + price_noise, 3), 2.50, 8.00))
    
    owners_val = real_row['previous_owners']
    if rng.random() < 0.20:
        new_row['previous_owners'] = int(np.clip(owners_val + 1, 1, 10))
    else:
        new_row['previous_owners'] = int(owners_val)
        
    # NEW COMPONENT INTEGRATION
    # 1. Engine Audio
    fault = rng.choice(FAULTS, p=FAULT_PROBS)
    confidence = float(rng.uniform(0.60, 1.00))
    if fault == "healthy":
        confidence = 1.0 # Healthy is generally high confidence
    
    new_row['fault_class'] = fault
    new_row['confidence'] = round(confidence, 2)
    
    # 2. Body Condition
    # Most cars are somewhat decent (60-100), some are bad
    body_score = int(rng.normal(85, 15))
    body_score = int(np.clip(body_score, 10, 100))
    new_row['body_score'] = body_score
    
    # 3. Apply Deductions to Target Price
    eng_deduction_lkr = get_engine_deduction(fault, confidence)
    body_deduction_lkr = get_body_deduction(body_score)
    
    total_deduction_million = (eng_deduction_lkr + body_deduction_lkr) / 1_000_000
    
    final_price = base_price_million - total_deduction_million
    final_price = max(0.5, final_price) # Floor at 500k
    
    new_row['price_million'] = round(final_price, 3)
    
    return new_row

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return

    df_real = pd.read_csv(INPUT_PATH)
    real_count = len(df_real)
    print(f"Loaded {real_count} real records.")
    
    if 'data_source' in df_real.columns:
        df_real = df_real.drop(columns=['data_source'])

    rng = np.random.default_rng(RANDOM_SEED)
    
    # We will treat ALL rows as needing the new synthetic features.
    # The real rows in alto_clean don't have fault_class or body_score. 
    # We will apply generate_holistic_row to EVERY record to give them these features.
    
    print("Synthesizing holistic features for all records...")
    
    all_data = []
    # First, make "perfect" versions of the real rows (healthy, score 100) to keep base truth
    for idx, row in df_real.iterrows():
        new_row = row.copy().to_dict()
        new_row['fault_class'] = 'healthy'
        new_row['confidence'] = 1.0
        new_row['body_score'] = 100
        new_row['data_source'] = 'real_augmented'
        all_data.append(new_row)
        
    # Then generate the rest
    synthetic_needed = TARGET_TOTAL_ROWS - len(all_data)
    sampled_indices = rng.choice(df_real.index, size=synthetic_needed, replace=True)
    
    for idx in sampled_indices:
        syn_row = generate_holistic_row(df_real.loc[idx], rng)
        syn_row['data_source'] = 'synthetic_holistic'
        all_data.append(syn_row)
        
    df_final = pd.DataFrame(all_data)
    df_final = df_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df_final)} total records.")
    print(f"Saved holistic dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

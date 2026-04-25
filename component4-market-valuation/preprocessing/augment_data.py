import pandas as pd
import numpy as np
import os

# Configuration
INPUT_PATH = "data/processed/alto_clean.csv"
OUTPUT_PATH = "data/processed/alto_augmented.csv"
TARGET_TOTAL_ROWS = 1000
RANDOM_SEED = 42

def generate_synthetic_row(real_row, rng):
    """
    Generates a synthetic row based on a real row with controlled variations.
    """
    new_row = real_row.copy().to_dict()
    
    # mileage_km: add Gaussian noise with std = 3% of the original value. Round to nearest 100. Clip between 5000 and 250000.
    mileage_val = real_row['mileage_km']
    mileage_noise = rng.normal(0, 0.03 * mileage_val)
    new_row['mileage_km'] = int(np.clip(round((mileage_val + mileage_noise) / 100) * 100, 5000, 250000))
    
    # price_million: add Gaussian noise with std = 1.5% of the original value. Round to 3 decimal places. Clip between 2.50 and 8.00.
    price_val = real_row['price_million']
    price_noise = rng.normal(0, 0.015 * price_val)
    new_row['price_million'] = float(np.clip(round(price_val + price_noise, 3), 2.50, 8.00))
    
    # previous_owners: with 20% probability add 1, otherwise keep same. Clip between 1 and 10. Convert to int.
    owners_val = real_row['previous_owners']
    if rng.random() < 0.20:
        new_row['previous_owners'] = int(np.clip(owners_val + 1, 1, 10))
    else:
        new_row['previous_owners'] = int(owners_val)
        
    # maf_year, reg_year, vehicle_age, reg_gap, power_shutters, power_mirrors, is_reconditioned: 
    # keep exactly the same as original row (already copied in new_row = real_row.copy())
    
    return new_row

def main():
    # Load the cleaned dataset
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file {INPUT_PATH} not found. Please ensure preprocessing/clean_data.py has been run.")
        return

    df_real = pd.read_csv(INPUT_PATH)
    real_count = len(df_real)
    print(f"Loaded {real_count} real records from {INPUT_PATH}")
    
    # Drop data_source column if it already exists to avoid conflicts
    if 'data_source' in df_real.columns:
        df_real = df_real.drop(columns=['data_source'])
    
    # Check if we already have enough records
    if real_count >= TARGET_TOTAL_ROWS:
        print("Already have enough records")
        df_real['data_source'] = 'real'
        df_real.to_csv(OUTPUT_PATH, index=False)
        return

    # Calculate synthetic rows needed
    synthetic_needed = TARGET_TOTAL_ROWS - real_count
    
    # Create numpy random generator
    rng = np.random.default_rng(RANDOM_SEED)
    
    # Generate synthetic rows: sample with replacement and apply generation function
    sampled_indices = rng.choice(df_real.index, size=synthetic_needed, replace=True)
    synthetic_data = [generate_synthetic_row(df_real.loc[idx], rng) for idx in sampled_indices]
    df_synthetic = pd.DataFrame(synthetic_data)
    
    # Add data_source column
    df_real['data_source'] = 'real'
    df_synthetic['data_source'] = 'synthetic'
    
    # Combine, shuffle, and reset index
    df_combined = pd.concat([df_real, df_synthetic], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    # Save to OUTPUT_PATH
    df_combined.to_csv(OUTPUT_PATH, index=False)
    
    # Validation check
    df_val = pd.read_csv(OUTPUT_PATH)
    val_total = len(df_val)
    
    # Statistics for validation
    real_mean_price = df_real['price_million'].mean()
    synth_mean_price = df_synthetic['price_million'].mean()
    price_diff = abs(real_mean_price - synth_mean_price) / real_mean_price
    
    real_mean_mileage = df_real['mileage_km'].mean()
    synth_mean_mileage = df_synthetic['mileage_km'].mean()
    mileage_diff = abs(real_mean_mileage - synth_mean_mileage) / real_mean_mileage
    
    if price_diff > 0.05:
        print(f"WARNING: Price mean difference is {price_diff:.1%}, which exceeds 5% threshold.")
    if mileage_diff > 0.05:
        print(f"WARNING: Mileage mean difference is {mileage_diff:.1%}, which exceeds 5% threshold.")
    
    if df_val.isnull().values.any():
        print("WARNING: Null values found in the augmented dataset!")
        
    # Year distribution for summary (maf_year)
    year_col = 'maf_year'
    real_years = df_real[year_col].value_counts().sort_index() if year_col in df_real.columns else pd.Series()
    synth_years = df_synthetic[year_col].value_counts().sort_index() if year_col in df_synthetic.columns else pd.Series()

    # Summary Print
    print("═══════════════════════════════════════════")
    print("DATA AUGMENTATION SUMMARY")
    print("═══════════════════════════════════════════")
    print(f"Real records:          {real_count}")
    print(f"Synthetic generated:   {synthetic_needed}")
    print(f"Total records:        {val_total}")
    print("")
    print("Price stats comparison:")
    print(f"  Real mean:           {real_mean_price:.2f}M LKR")
    print(f"  Synthetic mean:      {synth_mean_price:.2f}M LKR")
    print(f"  Difference:          {price_diff:.1%}  ← should be < 5%")
    print("")
    print("Mileage stats comparison:")
    print(f"  Real mean:           {real_mean_mileage:,.0f} km")
    print(f"  Synthetic mean:      {synth_mean_mileage:,.0f} km")
    print(f"  Difference:          {mileage_diff:.1%}  ← should be < 5%")
    print("")
    print("Year distribution (real vs synthetic):")
    all_years = sorted(set(real_years.index) | set(synth_years.index))
    for year in all_years:
        r_count = real_years.get(year, 0)
        s_count = synth_years.get(year, 0)
        print(f"  {year}: real={r_count}   synthetic={s_count}")
    print("")
    print(f"Saved to: {OUTPUT_PATH}")
    
    # Final check flag
    if val_total == TARGET_TOTAL_ROWS and price_diff <= 0.05 and mileage_diff <= 0.05 and not df_val.isnull().values.any():
        print("All validation checks passed ✓")
    
    print("═══════════════════════════════════════════")

if __name__ == "__main__":
    main()
    print("Data augmentation complete")

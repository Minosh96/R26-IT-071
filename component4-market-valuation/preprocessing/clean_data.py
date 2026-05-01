import pandas as pd
import numpy as np
import os
import re

# Configuration
INPUT_PATH = "data/raw/ALTO data_Updated.xlsx"
OUTPUT_PATH = "data/processed/alto_clean.csv"
MIN_YEAR = 2012
MAX_YEAR = 2017
MAX_PRICE_MILLION = 10.0
CURRENT_YEAR = 2026

def extract_owner_number(val):
    """Extracts the first digit from strings like '1st owner', '2nd owner' etc."""
    if pd.isna(val):
        return None
    match = re.search(r'\d', str(val))
    if match:
        return int(match.group())
    return None

def standardize_yes_no(val):
    """Standardizes yes/no variants to 1/0."""
    if pd.isna(val):
        return 0
    val_str = str(val).lower().strip()
    if val_str == 'yes':
        return 1
    elif val_str == 'no':
        return 0
    return 0

def clean_dataset():
    # Step 1 — Load data
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file {INPUT_PATH} not found.")
        return

    # Using openpyxl engine as requested
    df = pd.read_excel(INPUT_PATH, engine='openpyxl')
    total_loaded = len(df)
    print(f"Total rows loaded: {total_loaded}")
    print(f"Columns: {df.columns.tolist()}")

    # Step 2 — Remove price outliers
    mask_outlier = df['Price Listed (Million)'] > MAX_PRICE_MILLION
    dropped_outliers = mask_outlier.sum()
    df = df[~mask_outlier].copy()
    print(f"Dropped {dropped_outliers} price outliers.")

    # Step 3 — Filter year range
    df = df[(df['MAF Date'] >= MIN_YEAR) & (df['MAF Date'] <= MAX_YEAR)].copy()
    rows_after_year = len(df)
    print(f"Rows remaining after year filter: {rows_after_year}")

    # Step 4 — Clean previous owners column
    df['previous_owners'] = df['# of previous owners'].apply(extract_owner_number)
    median_owners = df['previous_owners'].median()
    df['previous_owners'] = df['previous_owners'].fillna(median_owners).astype(int)

    # Step 5 — Clean BN/ R column
    # BN -> 0 (Brand New), R -> 1 (Reconditioned)
    df['is_reconditioned'] = df['BN/ R'].map({'BN': 0, 'R': 1}).fillna(0).astype(int)

    # Step 6 — Clean power shutters
    df['power_shutters'] = df['Power shutters'].apply(standardize_yes_no)

    # Step 7 — Clean power mirrors
    df['power_mirrors'] = df['Power mirrors'].apply(standardize_yes_no)

    # Step 8 — Add vehicle age
    df['vehicle_age'] = CURRENT_YEAR - df['MAF Date']

    # Step 9 — Add registration gap
    df['reg_gap'] = df['Reg Date '] - df['MAF Date']
    df['reg_gap'] = df['reg_gap'].clip(lower=0)

    # Step 10 — Fill missing mileage
    # Fill null mileage with median of the same year group, then overall median
    df['MileAge (km)'] = df.groupby('MAF Date')['MileAge (km)'].transform(lambda x: x.fillna(x.median()))
    df['MileAge (km)'] = df['MileAge (km)'].fillna(df['MileAge (km)'].median())

    # Step 11 — Rename columns
    df = df.rename(columns={
        'MAF Date': 'maf_year',
        'Reg Date ': 'reg_year',
        'MileAge (km)': 'mileage_km',
        'Price Listed (Million)': 'price_million'
    })

    # Step 12 — Keep only final columns
    final_columns = [
        'maf_year', 'reg_year', 'vehicle_age', 'reg_gap', 'mileage_km', 
        'previous_owners', 'is_reconditioned', 'power_shutters', 'power_mirrors', 'price_million'
    ]
    df = df[final_columns].copy()

    # Step 13 — Final validation
    for col in df.columns:
        if df[col].isnull().any():
            print(f"WARNING: Column '{col}' still has null values!")

    # Step 14 — Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # Step 15 — Print summary
    print("═══════════════════════════════════")
    print("DATA CLEANING SUMMARY")
    print("═══════════════════════════════════")
    print(f"Rows loaded:          {total_loaded}")
    print(f"After outlier removal: {total_loaded - dropped_outliers}")
    print(f"After year filter:    {rows_after_year}")
    print(f"Final rows:           {len(df)}")
    print(f"Columns:              {len(df.columns)}")
    print("")
    print("Price stats (million LKR):")
    print(f"  Min:    {df['price_million'].min():.2f}")
    print(f"  Max:    {df['price_million'].max():.2f}")
    print(f"  Mean:   {df['price_million'].mean():.2f}")
    print(f"  Median: {df['price_million'].median():.2f}")
    print("")
    print("Mileage stats (km):")
    print(f"  Min:    {df['mileage_km'].min():.0f}")
    print(f"  Max:    {df['mileage_km'].max():.0f}")
    print(f"  Mean:   {df['mileage_km'].mean():.0f}")
    print("")
    print("Year distribution:")
    year_counts = df['maf_year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count} vehicles")
    print("")
    print(f"Saved to: {OUTPUT_PATH}")
    print("═══════════════════════════════════")

if __name__ == "__main__":
    clean_dataset()
    print("Data cleaning complete")

"""
Clean the real Suzuki Alto listings (no synthetic rows).

This dataset is the ONLY training data for the market price model. Vehicle
condition (engine fault, body damage) is NOT learned from listings, because
listings do not record it. Condition is priced separately from the repair
cost survey in data/repair_costs.json (see inference/valuate.py).
"""
import os
import re
import json
import pandas as pd

INPUT_PATH = "data/raw/ALTO_data_v2.xlsx"
OUTPUT_PATH = "data/processed/alto_real.csv"
LOG_PATH = "data/processed/alto_real_cleaning_log.json"
MIN_YEAR = 2010
MAX_YEAR = 2018
MAX_PRICE_MILLION = 10.0   # 36.5 / 34.9 are data-entry errors (an Alto is ~3-5 Mn)
REFERENCE_YEAR = 2026      # listed_months is counted from Dec 2025 (1 = Jan 2026)


def owner_number(val):
    """'2nd owner', '3 rd Owner', '2d owner' -> 2"""
    match = re.search(r"\d", str(val))
    return int(match.group()) if match else None


def yes_no(val):
    return 1 if str(val).strip().lower() == "yes" else 0


def listed_months(val):
    """'2026/10' or '2024-01' -> months relative to Dec 2025."""
    match = re.search(r"(\d{4})\D(\d{1,2})", str(val))
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    return (year - REFERENCE_YEAR) * 12 + month


def clean_dataset():
    raw = pd.read_excel(INPUT_PATH, sheet_name="Data Table", engine="openpyxl")
    log = {"rows_loaded": len(raw)}

    df = pd.DataFrame({
        "vehicle_id": raw["Vehicle ID"],
        "plate": raw["Reg No"].astype(str).str.replace(" ", "").str.upper(),
        "maf_year": raw["MAF Date"],
        "reg_year": raw["Reg Date "],
        "mileage_km": raw["MileAge (km)"],
        "previous_owners": raw["# of previous owners"].apply(owner_number),
        "is_reconditioned": (raw["BN/ R"].astype(str).str.strip().str.upper() == "R").astype(int),
        "power_shutters": raw["Power shutters"].apply(yes_no),
        "power_mirrors": raw["Power mirrors"].apply(yes_no),
        "listed_months": raw["Year/month listed to sell"].apply(listed_months),
        "price_million": raw["Price Listed (Million)"],
    })

    outliers = df["price_million"] > MAX_PRICE_MILLION
    log["dropped_price_outliers"] = df.loc[outliers, "vehicle_id"].tolist()
    df = df[~outliers]

    in_scope = df["maf_year"].between(MIN_YEAR, MAX_YEAR)
    log["dropped_out_of_year_range"] = int((~in_scope).sum())
    df = df[in_scope].copy()

    # The same car listed twice (same full plate) is kept once: the latest listing.
    full_plate = df["plate"].str.contains(r"\d{4}")
    relisted = full_plate & df.duplicated("plate", keep="last")
    log["dropped_relistings"] = df.loc[relisted, "vehicle_id"].tolist()
    df = df[~relisted].copy()

    log["filled_missing"] = {
        "previous_owners": int(df["previous_owners"].isna().sum()),
        "mileage_km": int(df["mileage_km"].isna().sum()),
        "listed_months": int(df["listed_months"].isna().sum()),
    }
    df["previous_owners"] = df["previous_owners"].fillna(df["previous_owners"].median()).astype(int)
    df["mileage_km"] = df.groupby("maf_year")["mileage_km"].transform(lambda x: x.fillna(x.median()))
    df["listed_months"] = df["listed_months"].fillna(df["listed_months"].mode()[0]).astype(int)

    # A car is grouped by plate when the full plate is known, so the same car
    # never appears in both the training and test folds.
    df["group"] = df["plate"].where(df["plate"].str.contains(r"\d{4}"), df["vehicle_id"])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    log["final_rows"] = len(df)
    log["year_counts"] = {int(k): int(v) for k, v in df["maf_year"].value_counts().sort_index().items()}
    log["listing_month_counts"] = {int(k): int(v) for k, v in df["listed_months"].value_counts().sort_index().items()}
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    print(f"Rows loaded:        {log['rows_loaded']}")
    print(f"Price outliers:     {log['dropped_price_outliers']}")
    print(f"Outside {MIN_YEAR}-{MAX_YEAR}:  {log['dropped_out_of_year_range']}")
    print(f"Re-listings:        {log['dropped_relistings']}")
    print(f"Final real rows:    {len(df)}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    clean_dataset()

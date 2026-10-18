"""
Market check: compare the app's fair price with cars that are for sale right now.

1. Copy validation/market_check_template.csv to validation/market_check.csv and add
   10-20 Suzuki Altos currently listed (riyasewana.lk, ikman.lk, a dealer ...).
   accepted_price_million is optional: the price a seller or dealer said they would
   actually accept, if you asked.
2. Run from the component folder:
       python validation/run_market_check.py
3. Use validation/market_check_results.csv and the printed table in the slides.

Also writes validation/mechanic_review.csv: the surveyed repair cost ranges with
empty columns for a mechanic to fill in ("do these ranges look right?").
"""
import csv
import json
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))
from inference.valuate import load_all, valuate  # noqa: E402

INPUT = "validation/market_check.csv"
OUTPUT = "validation/market_check_results.csv"
MECHANIC_SHEET = "validation/mechanic_review.csv"
VEHICLE_FIELDS = ["maf_year", "reg_year", "mileage_km", "previous_owners",
                  "is_reconditioned", "power_shutters", "power_mirrors"]


def write_mechanic_sheet():
    with open("data/repair_costs.json") as f:
        costs = json.load(f)
    rows = []
    for group in ("engine_faults", "body_damage_categories", "body_parts", "electrical"):
        for name, data in costs.get(group, {}).items():
            if isinstance(data, dict) and "min" in data:
                rows.append({"group": group, "item": name, "survey_min_lkr": data["min"],
                             "survey_max_lkr": data["max"], "mechanic_min_lkr": "",
                             "mechanic_max_lkr": "", "mechanic_comment": ""})
    with open(MECHANIC_SHEET, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Mechanic review sheet: {MECHANIC_SHEET} ({len(rows)} repair items)")


def main():
    write_mechanic_sheet()
    if not pathlib.Path(INPUT).exists():
        print(f"\nNo {INPUT} yet. Copy market_check_template.csv to market_check.csv, add the cars, and run again.")
        return

    assets = load_all()
    with open(INPUT, newline="") as f:
        cars = [row for row in csv.DictReader(f) if row.get("car_id") and not row["car_id"].startswith("EXAMPLE")]

    results = []
    for car in cars:
        vehicle = {k: int(float(car[k])) for k in VEHICLE_FIELDS}
        asking = float(car["asking_price_million"])
        # Listings are judged as seen in the advert: condition and VIN not inspected
        r = valuate(vehicle, asking, vin_status="original", preloaded_assets=assets)
        row = {"car_id": car["car_id"], "source": car.get("source", ""),
               "asking_lkr": int(asking * 1e6), "status": r["status"]}
        if r["status"] == "success":
            fair, lo, hi = r["fair_value_lkr"], r["fair_low_lkr"], r["fair_high_lkr"]
            row.update({"fair_value_lkr": fair, "fair_low_lkr": lo, "fair_high_lkr": hi,
                        "verdict": r["verdict"], "asking_inside_range": lo <= row["asking_lkr"] <= hi,
                        "error_vs_asking_lkr": fair - row["asking_lkr"]})
            if car.get("accepted_price_million"):
                accepted = int(float(car["accepted_price_million"]) * 1e6)
                row.update({"accepted_lkr": accepted, "error_vs_accepted_lkr": fair - accepted,
                            "accepted_inside_range": lo <= accepted <= hi})
        else:
            row["verdict"] = r.get("message", "")
        results.append(row)

    fields = []
    for row in results:
        fields += [k for k in row if k not in fields]
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    valued = [r for r in results if r["status"] == "success"]
    print(f"\n{'Car':12s} {'Asking':>11s} {'App fair':>11s} {'App range':>23s}  Verdict")
    for r in results:
        if r["status"] == "success":
            print(f"{r['car_id']:12s} {r['asking_lkr']:11,} {r['fair_value_lkr']:11,} "
                  f"{r['fair_low_lkr']:>11,}-{r['fair_high_lkr']:<11,} {r['verdict']}")
        else:
            print(f"{r['car_id']:12s} {r['asking_lkr']:11,}  not valued: {r['verdict']}")
    if valued:
        errors = [abs(r["error_vs_asking_lkr"]) for r in valued]
        pct = [abs(r["error_vs_asking_lkr"]) / r["asking_lkr"] * 100 for r in valued]
        inside = sum(r["asking_inside_range"] for r in valued)
        print(f"\nCars valued: {len(valued)} of {len(results)}")
        print(f"Average difference from asking price: LKR {sum(errors)/len(errors):,.0f} ({sum(pct)/len(pct):.1f}%)")
        print(f"Asking price inside the app's fair range: {inside} of {len(valued)}")
        accepted = [r for r in valued if "accepted_lkr" in r]
        if accepted:
            err = sum(abs(r["error_vs_accepted_lkr"]) for r in accepted) / len(accepted)
            print(f"Average difference from accepted price: LKR {err:,.0f} ({len(accepted)} cars)")
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()

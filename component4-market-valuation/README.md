# Component 4: Market Valuation

Gives a used Suzuki Alto buyer a fair price range, a verdict (good deal / fair price / overpriced),
a recommended offer and a negotiation range, with a line-by-line explanation of the price.

Run instructions: [RUNNING.md](../RUNNING.md). API docs: `http://localhost:5004/apidocs/`.

## How a price is calculated

```
fair value = ML market price              trained on real listings only
           - engine repair cost            engine health colour (component 3) x surveyed repair cost
           - body repair cost              body score band (component 2) x surveyed repair cost
           - VIN risk discount             only when the VIN needs review (component 1)
```

**Why it is split.** Listings record year, mileage, owners and features, but not engine or body
condition, so no model can learn condition from them. The market price is machine learning; the
condition adjustment comes from our field survey of 5 sellers and repair centres
([data/repair_costs.json](data/repair_costs.json)). Each deduction is shown separately in the app.

| Input | How it is used |
|---|---|
| Year, mileage, owners, registration gap, power shutters/mirrors | Market price model ([models/market_model.py](models/market_model.py)) |
| Engine health score (MHS) | green: no deduction, amber: half the typical engine repair, red: the full typical repair. The fault name is only shown as a possible cause, because the engine component is reliable at healthy vs faulty but weaker at naming the fault. |
| Body score | 100 none, 90-99 minor, 60-89 moderate, below 60 severe (same bands as component 2's condition labels), priced from the survey |
| VIN status | altered: no price, "do not buy". Needs review: 10% risk discount and never shown as a good deal. Not checked: warning only. |

**Verdict.** The fair range covers 80% of real listings around the model's prediction
(-5.7% / +6.1%, measured on unseen listings). Above the range is overpriced, below is a good deal.

**Explanation.** For each factor the car's value is swapped for a typical listing's value and the
model is asked again, e.g. "-LKR 78,000 for 160,000 km (typical: 112,000 km)".

**Direction rules.** The model may never raise the price for more mileage, more owners or an older
car (monotonic constraints). Without them it learned a small, spurious "more owners, higher price" effect.

## Scope

Suzuki Alto, manufactured 2010-2018, 30,000-252,100 km, 1-8 previous owners. Other cars are
refused with "this car can't be valued reliably" instead of being given a confident wrong price.
Adding a model (e.g. Wagon R) means collecting its listings in the same spreadsheet format and retraining.

## Evaluation (real listings only, no synthetic data)

553 real listings after cleaning (2 price typos, 23 pre-2010 cars and 5 re-listings of the same car removed).
Error is average absolute difference from the real listed price.

| Model | 20% hold-out | Grouped 5-fold CV | Train Jan-Mar, test Oct |
|---|---|---|---|
| Year-average baseline | LKR 149,292 (4.00%) | LKR 154,586 (4.10%) | LKR 215,044 (5.71%) |
| Linear regression | LKR 145,010 (3.88%) | LKR 144,445 (3.84%) | LKR 179,400 (4.59%) |
| Gradient boosting | LKR 131,942 (3.53%) | LKR 143,719 (3.82%) | LKR 234,215 (6.04%) |
| **Hybrid (chosen)** | LKR 144,638 (3.86%) | **LKR 140,799 (3.76%)** | LKR 215,207 (5.57%) |

The hybrid is chosen because it has the lowest error in cross-validation (the most reliable of the
three tests), follows the direction rules, and handles the price trend over time. Tree models do
slightly better on the single 20% split but extrapolate poorly to new months.

Full numbers: `models/saved/market_model_report.json`. Figures: `thesis/figures/v2_*.png`.

## Keeping prices current

Asking prices rose about 5% between January and October 2026. A model trained on January-March
listings had 5.6% error on October listings; adding just 40 October listings brought it back to 3.6%.

Monthly refresh:
1. Add new listings to `data/raw/ALTO_data_v2.xlsx`, with the "Year/month listed to sell" column filled.
2. `python preprocessing/clean_real_data.py`
3. `python models/train_market_model.py`
4. `pytest`, then commit `models/saved/market_model*`.

Re-check the repair cost survey every quarter. The API warns (`market_data_warning`) when the newest
listings are more than 3 months old.

## Market check (practical usage)

1. Copy `validation/market_check_template.csv` to `validation/market_check.csv` and add 10-20 Altos
   currently for sale (and the price the seller would accept, if you ask).
2. `python validation/run_market_check.py` prints the comparison table and summary.
3. The same script writes `validation/mechanic_review.csv` so a mechanic can confirm the repair cost ranges.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The GitHub Actions workflow `.github/workflows/component4-valuation.yml` runs the tests and starts the
API under Gunicorn on every change to this folder. Library versions are pinned in `requirements.txt`
because the saved model is a pickled scikit-learn object.

"""
Market price model shared by training and inference.

The model only uses facts that appear in a real listing. Condition is handled
outside the model, from the repair cost survey.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

CURRENT_YEAR_BASE = 2026  # listed_months is counted from Dec 2025

FEATURES = [
    "maf_year", "age_at_listing", "mileage_km", "km_per_year", "previous_owners",
    "reg_gap", "is_reconditioned", "power_shutters", "power_mirrors", "listed_months",
]

# Domain knowledge: the direction each factor is allowed to move the price.
#  +1 price can only go up as the value rises, -1 only down, 0 free (learned from data).
# Without this the model can pick up noise, e.g. "more previous owners -> higher price".
DIRECTIONS = {
    "maf_year": +1, "age_at_listing": -1, "mileage_km": -1, "km_per_year": -1,
    "previous_owners": -1, "reg_gap": 0, "is_reconditioned": 0,
    "power_shutters": +1, "power_mirrors": +1, "listed_months": 0,
}


def add_features(df):
    """Derive model features from cleaned listing columns (works for 1 or many rows)."""
    df = df.copy()
    df["reg_gap"] = (df["reg_year"] - df["maf_year"]).clip(lower=0)
    # Training: age when the listing was collected. Inference: age today (age_ref_months),
    # while listed_months stays at the latest market month the model has seen.
    age_months = df["age_ref_months"] if "age_ref_months" in df else df["listed_months"]
    df["age_at_listing"] = (CURRENT_YEAR_BASE + age_months / 12) - df["maf_year"]
    df["km_per_year"] = df["mileage_km"] / df["age_at_listing"].clip(lower=1)
    return df[FEATURES]


class HybridRegressor(BaseEstimator, RegressorMixin):
    """
    A sign-constrained linear model captures the overall trend (age, mileage,
    market drift over time) and can extrapolate to the newest month; gradient
    boosting with monotonic constraints then learns non-linear corrections from
    the residuals. Both parts respect DIRECTIONS.
    """

    def __init__(self, max_iter=200, learning_rate=0.03, max_depth=2, min_samples_leaf=10, random_state=42):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _linear_design(self, X):
        # positive=True forces every coefficient >= 0. Constrained features are
        # multiplied by their direction; free features enter as +x and -x so their
        # net coefficient can take either sign.
        Z = self.scaler_.transform(X)
        cols = []
        for i, feature in enumerate(FEATURES):
            sign = DIRECTIONS[feature]
            cols.append(Z[:, i] * (sign if sign else 1))
            if sign == 0:
                cols.append(-Z[:, i])
        return np.column_stack(cols)

    def fit(self, X, y):
        X = X[FEATURES]
        self.scaler_ = StandardScaler().fit(X)
        self.linear_ = LinearRegression(positive=True).fit(self._linear_design(X), y)
        self.trees_ = HistGradientBoostingRegressor(
            max_iter=self.max_iter, learning_rate=self.learning_rate, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, random_state=self.random_state,
            monotonic_cst=[DIRECTIONS[f] for f in FEATURES],
        ).fit(X, y - self.linear_.predict(self._linear_design(X)))
        return self

    def predict(self, X):
        X = X[FEATURES]
        return self.linear_.predict(self._linear_design(X)) + self.trees_.predict(X)

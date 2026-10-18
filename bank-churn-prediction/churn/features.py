"""Feature engineering, expressed as a scikit-learn transformer.

Mirrors the notebook's feature set. Unlike the notebook, anything learned from
data (the balance and salary medians behind IsHighValue) is learned on the
training split only, so there is no leakage into the test set.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import RAW_FEATURES

FEATURE_ORDER = [
    "CreditScore", "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary", "BalanceToSalaryRatio",
    "HasBalance", "EngagementScore", "IsHighValue", "Geo_Germany", "Geo_Spain",
    "AgeGroup_Middle", "AgeGroup_Senior", "AgeGroup_Elderly",
    "TenureCategory_Moderate", "TenureCategory_Long",
]

_AGE_BINS = [0, 30, 45, 60, np.inf]
_AGE_LABELS = ["Young", "Middle", "Senior", "Elderly"]
_TENURE_BINS = [-1, 2, 5, np.inf]
_TENURE_LABELS = ["New", "Moderate", "Long"]


class ChurnFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = self._check(X)
        self.balance_median_ = float(X["Balance"].median())
        self.salary_median_ = float(X["EstimatedSalary"].median())
        return self

    def transform(self, X):
        X = self._check(X)
        out = pd.DataFrame(index=X.index)
        out["CreditScore"] = X["CreditScore"].astype(float)
        out["Gender"] = (X["Gender"].astype(str).str.lower() == "male").astype(int)
        out["Age"] = X["Age"].astype(float)
        out["Tenure"] = X["Tenure"].astype(float)
        out["Balance"] = X["Balance"].astype(float)
        out["NumOfProducts"] = X["NumOfProducts"].astype(float)
        out["HasCrCard"] = X["HasCrCard"].astype(int)
        out["IsActiveMember"] = X["IsActiveMember"].astype(int)
        out["EstimatedSalary"] = X["EstimatedSalary"].astype(float)
        out["BalanceToSalaryRatio"] = X["Balance"] / (X["EstimatedSalary"] + 1)
        out["HasBalance"] = (X["Balance"] > 0).astype(int)
        out["EngagementScore"] = (
            X["NumOfProducts"] * 0.4 + X["IsActiveMember"] * 0.3 + X["HasCrCard"] * 0.3
        ).astype(float)
        out["IsHighValue"] = (
            (X["Balance"] > self.balance_median_) & (X["EstimatedSalary"] > self.salary_median_)
        ).astype(int)
        geo = X["Geography"].astype(str).str.title()
        out["Geo_Germany"] = (geo == "Germany").astype(int)
        out["Geo_Spain"] = (geo == "Spain").astype(int)
        age = pd.cut(X["Age"], bins=_AGE_BINS, labels=_AGE_LABELS).astype(str)
        for label in ("Middle", "Senior", "Elderly"):
            out[f"AgeGroup_{label}"] = (age == label).astype(int)
        ten = pd.cut(X["Tenure"], bins=_TENURE_BINS, labels=_TENURE_LABELS).astype(str)
        for label in ("Moderate", "Long"):
            out[f"TenureCategory_{label}"] = (ten == label).astype(int)
        return out[FEATURE_ORDER]

    @staticmethod
    def _check(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ChurnFeatures expects a pandas DataFrame with raw columns")
        missing = [c for c in RAW_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return X

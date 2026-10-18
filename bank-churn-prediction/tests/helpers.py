"""Synthetic data used ONLY to exercise the code in tests. Never used by the app."""
import numpy as np
import pandas as pd


def make_synthetic(n=3000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    geo = rng.choice(["France", "Germany", "Spain"], n, p=[0.5, 0.25, 0.25])
    age = np.clip(rng.normal(39, 10.5, n), 18, 92).round().astype(int)
    active = rng.integers(0, 2, n)
    products = rng.choice([1, 2, 3, 4], n, p=[0.5, 0.45, 0.04, 0.01])
    balance = np.where(rng.random(n) < 0.36, 0.0, rng.normal(120000, 30000, n).clip(1000))
    logit = -3.2 + 0.06 * (age - 30) + 0.8 * (geo == "Germany") - 1.0 * active
    exited = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "RowNumber": np.arange(1, n + 1),
        "CustomerId": 15_600_000 + np.arange(n),
        "Surname": "Test",
        "CreditScore": rng.normal(650, 96, n).clip(350, 850).round().astype(int),
        "Geography": geo,
        "Gender": rng.choice(["Female", "Male"], n),
        "Age": age,
        "Tenure": rng.integers(0, 11, n),
        "Balance": balance.round(2),
        "NumOfProducts": products,
        "HasCrCard": rng.integers(0, 2, n),
        "IsActiveMember": active,
        "EstimatedSalary": rng.uniform(10, 200000, n).round(2),
        "Exited": exited,
    })

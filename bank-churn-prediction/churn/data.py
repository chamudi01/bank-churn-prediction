"""Loading, cleaning and validating data."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BINARY_INPUTS, CATEGORICAL_INPUTS, ID_COLUMNS, NUMERIC_INPUTS,
    RAW_FEATURES, TARGET, VALID_RANGES,
)


def load_raw(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place Churn_Modelling.csv in data/raw/ "
            "or pass --data."
        )
    df = pd.read_csv(path)
    missing = [c for c in RAW_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning rules carried over from the notebook.

    - drop identifier columns
    - drop rows with missing required values
    - drop CreditScore outliers outside 1.5 x IQR
    """
    out = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])
    out = out.dropna(subset=RAW_FEATURES + [TARGET])
    q1, q3 = out["CreditScore"].quantile([0.25, 0.75])
    iqr = q3 - q1
    keep = out["CreditScore"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return out.loc[keep].reset_index(drop=True)


def validate_scoring_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a typed copy containing only the model inputs, or raise ValueError
    with a readable list of problems."""
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    X = df[RAW_FEATURES].copy()
    problems = []

    for col in NUMERIC_INPUTS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        n_bad = int(X[col].isna().sum())
        if n_bad:
            problems.append(f"{col}: {n_bad} missing or non-numeric value(s)")

    for col, allowed in CATEGORICAL_INPUTS.items():
        X[col] = X[col].astype(str).str.strip().str.title()
        bad = sorted(set(X[col]) - set(allowed))
        if bad:
            shown = ", ".join(bad[:5])
            problems.append(f"{col}: unrecognised value(s) {shown} (expected {', '.join(allowed)})")

    for col, (lo, hi) in VALID_RANGES.items():
        s = X[col].dropna()
        if lo is not None and (s < lo).any():
            problems.append(f"{col}: {int((s < lo).sum())} value(s) below {lo}")
        if hi is not None and (s > hi).any():
            problems.append(f"{col}: {int((s > hi).sum())} value(s) above {hi}")

    for col in BINARY_INPUTS:
        s = X[col].dropna()
        if (~s.isin([0, 1])).any():
            problems.append(f"{col}: must be 0 or 1")

    if problems:
        raise ValueError("\n".join(problems))
    return X.reset_index(drop=True)

"""Loading artifacts and scoring. No Streamlit imports: usable from scripts and tests."""
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, RAW_FEATURES, RISK_BAND_EDGES, RISK_BAND_LABELS

REQUIRED_FILES = ("models.joblib", "metadata.json", "reference.json",
                  "test_predictions.csv", "importance.csv")


class ArtifactsMissing(FileNotFoundError):
    pass


@dataclass(frozen=True)
class Bundle:
    models: dict
    metadata: dict
    reference: dict
    predictions: pd.DataFrame
    importance: pd.DataFrame


def load_bundle(directory=ARTIFACT_DIR) -> Bundle:
    d = Path(directory)
    missing = [f for f in REQUIRED_FILES if not (d / f).exists()]
    if missing:
        raise ArtifactsMissing(f"Missing artifact file(s) in {d}: {', '.join(missing)}")
    return Bundle(
        models=joblib.load(d / "models.joblib"),
        metadata=json.loads((d / "metadata.json").read_text()),
        reference=json.loads((d / "reference.json").read_text()),
        predictions=pd.read_csv(d / "test_predictions.csv"),
        importance=pd.read_csv(d / "importance.csv"),
    )


def score(pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(X[RAW_FEATURES])[:, 1]


def risk_band(prob):
    """Map probability (scalar or array) to Low / Medium / High."""
    idx = np.digitize(prob, RISK_BAND_EDGES)
    labels = np.array(RISK_BAND_LABELS, dtype=object)
    return labels[idx] if np.ndim(idx) else str(labels[int(idx)])


def baseline_row(reference: dict) -> dict:
    row = {c: v["median"] for c, v in reference["numeric"].items()}
    row.update({c: v["mode"] for c, v in reference["categorical"].items()})
    return row


def attribute_effects(pipeline, row: dict, reference: dict):
    """One-at-a-time effect of each attribute versus a typical customer.

    For every attribute the customer's value is swapped for the training-set
    median (numeric) or mode (categorical); the effect is the resulting change
    in predicted probability. Positive means the attribute raises churn risk.
    Interactions are ignored, so effects need not sum to the total difference.
    """
    base = baseline_row(reference)
    rows = [dict(row)]
    for feature in RAW_FEATURES:
        swapped = dict(row)
        swapped[feature] = base[feature]
        rows.append(swapped)
    probs = pipeline.predict_proba(pd.DataFrame(rows)[RAW_FEATURES])[:, 1]
    effects = pd.DataFrame({
        "feature": RAW_FEATURES,
        "value": [row[f] for f in RAW_FEATURES],
        "typical": [base[f] for f in RAW_FEATURES],
        "effect": probs[0] - probs[1:],
    })
    return float(probs[0]), effects


def response_curve(pipeline, reference: dict, feature: str, points: int = 60) -> pd.DataFrame:
    """Predicted probability for the reference customer as one attribute varies."""
    base = baseline_row(reference)
    if feature in reference["categorical"]:
        grid = reference["categorical"][feature]["values"]
    else:
        info = reference["numeric"][feature]
        lo, hi = info["min"], info["max"]
        if feature in ("Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember"):
            grid = list(range(int(lo), int(hi) + 1))
        else:
            grid = list(np.linspace(lo, hi, points))
    rows = []
    for v in grid:
        r = dict(base)
        r[feature] = v
        rows.append(r)
    probs = pipeline.predict_proba(pd.DataFrame(rows)[RAW_FEATURES])[:, 1]
    return pd.DataFrame({"value": grid, "probability": probs})

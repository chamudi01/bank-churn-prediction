"""Train candidate models and write artifacts for the dashboard.

Usage:
    python -m churn.train
    python -m churn.train --data path/to/Churn_Modelling.csv --model "Decision Tree"
"""
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .config import (
    ARTIFACT_DIR, CATEGORICAL_INPUTS, CV_FOLDS, DEFAULT_DATA_PATH, NUMERIC_INPUTS,
    RANDOM_STATE, RAW_FEATURES, TARGET, TEST_SIZE,
)
from .data import clean, load_raw
from .evaluate import metrics_at
from .modeling import build_pipeline, model_factories


def build_reference(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    numeric = {
        c: {
            "min": float(X_train[c].min()), "max": float(X_train[c].max()),
            "median": float(X_train[c].median()), "mean": float(X_train[c].mean()),
        }
        for c in NUMERIC_INPUTS
    }
    categorical = {
        c: {"values": list(vals), "mode": str(X_train[c].mode().iloc[0])}
        for c, vals in CATEGORICAL_INPUTS.items()
    }
    return {"numeric": numeric, "categorical": categorical,
            "churn_rate": float(y_train.mean())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--out", default=str(ARTIFACT_DIR))
    parser.add_argument("--model", default=None,
                        help="Force the dashboard's default model (otherwise best CV ROC-AUC).")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.data)
    df = clean(raw)
    print(f"Rows: {len(raw)} raw, {len(df)} after cleaning ({len(raw) - len(df)} removed)")

    X, y = df[RAW_FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models, summary, preds, importances = {}, {}, {"y_true": y_test.to_numpy()}, []

    for name, make in model_factories().items():
        pipe = build_pipeline(make())
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        models[name] = pipe
        preds[name] = prob
        summary[name] = {
            "cv_roc_auc_mean": float(cv_auc.mean()),
            "cv_roc_auc_std": float(cv_auc.std()),
            "test": metrics_at(y_test, prob, 0.5),
        }
        imp = permutation_importance(
            pipe, X_test, y_test, scoring="roc_auc", n_repeats=10,
            random_state=RANDOM_STATE,
        )
        importances.append(pd.DataFrame({
            "model": name, "feature": RAW_FEATURES,
            "importance_mean": imp.importances_mean, "importance_std": imp.importances_std,
        }))
        print(f"{name:28s} CV AUC {cv_auc.mean():.4f} | test AUC {summary[name]['test']['roc_auc']:.4f}")

    if args.model:
        if args.model not in models:
            sys.exit(f"Unknown model '{args.model}'. Choose from: {', '.join(models)}")
        selected = args.model
    else:
        selected = max(summary, key=lambda n: summary[n]["cv_roc_auc_mean"])
    print(f"Default model: {selected}")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_file": Path(args.data).name,
        "rows_raw": int(len(raw)), "rows_modelled": int(len(df)),
        "rows_removed": int(len(raw) - len(df)),
        "churn_rate": float(y.mean()),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "selected_model": selected,
        "selection_rule": f"highest mean {CV_FOLDS}-fold ROC-AUC on the training split"
        if not args.model else "set manually with --model",
        "models": summary,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }

    joblib.dump(models, out_dir / "models.joblib")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "reference.json").write_text(json.dumps(build_reference(X_train, y_train), indent=2))
    pd.DataFrame(preds).to_csv(out_dir / "test_predictions.csv", index=False)
    pd.concat(importances).to_csv(out_dir / "importance.csv", index=False)
    print(f"Artifacts written to {out_dir}")


if __name__ == "__main__":
    main()

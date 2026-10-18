import numpy as np
import pandas as pd

from churn.config import RAW_FEATURES, TARGET
from churn.data import clean, validate_scoring_frame
from churn.features import FEATURE_ORDER, ChurnFeatures
from churn.modeling import build_pipeline, model_factories
from churn.predict import attribute_effects, load_bundle, risk_band, response_curve
from churn.train import main as train_main
from tests.helpers import make_synthetic


def test_clean_drops_ids_and_keeps_schema():
    df = clean(make_synthetic())
    assert "CustomerId" not in df.columns
    assert set(RAW_FEATURES + [TARGET]) <= set(df.columns)


def test_feature_names_match_notebook_order():
    X = clean(make_synthetic())[RAW_FEATURES]
    out = ChurnFeatures().fit(X).transform(X)
    assert list(out.columns) == FEATURE_ORDER
    assert len(FEATURE_ORDER) == 20
    assert not out.isna().any().any()


def test_medians_learned_from_training_data_only():
    df = clean(make_synthetic())
    train, other = df.iloc[:1000][RAW_FEATURES], df.iloc[1000:][RAW_FEATURES]
    f = ChurnFeatures().fit(train)
    f.transform(other)
    assert f.balance_median_ == float(train["Balance"].median())


def test_every_model_scores_in_unit_interval():
    df = clean(make_synthetic())
    X, y = df[RAW_FEATURES], df[TARGET]
    for make in model_factories().values():
        p = build_pipeline(make()).fit(X, y).predict_proba(X)[:, 1]
        assert ((p >= 0) & (p <= 1)).all()


def test_validation_reports_problems():
    bad = make_synthetic(5)
    bad.loc[0, "Geography"] = "Mars"
    bad.loc[1, "Age"] = 5
    bad.loc[2, "HasCrCard"] = 3
    try:
        validate_scoring_frame(bad)
    except ValueError as e:
        msg = str(e)
        assert "Geography" in msg and "Age" in msg and "HasCrCard" in msg
    else:
        raise AssertionError("expected ValueError")


def test_risk_band_edges():
    assert risk_band(0.05) == "Low"
    assert risk_band(0.20) == "Medium"
    assert risk_band(0.50) == "High"
    assert list(risk_band(np.array([0.1, 0.3, 0.9]))) == ["Low", "Medium", "High"]


def test_train_end_to_end(tmp_path):
    csv = tmp_path / "data.csv"
    make_synthetic(2500).to_csv(csv, index=False)
    out = tmp_path / "artifacts"
    train_main(["--data", str(csv), "--out", str(out)])
    bundle = load_bundle(out)
    name = bundle.metadata["selected_model"]
    pipe = bundle.models[name]
    row = {k: v["median"] for k, v in bundle.reference["numeric"].items()}
    row.update({k: v["mode"] for k, v in bundle.reference["categorical"].items()})
    p, effects = attribute_effects(pipe, row, bundle.reference)
    assert 0 <= p <= 1 and len(effects) == len(RAW_FEATURES)
    # swapping a typical customer's attributes for typical values changes nothing
    assert np.allclose(effects["effect"], 0)
    curve = response_curve(pipe, bundle.reference, "Age")
    assert curve["probability"].between(0, 1).all()
    assert set(bundle.importance["model"]) == set(bundle.models)

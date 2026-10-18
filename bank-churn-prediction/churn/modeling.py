"""Model definitions and the end-to-end pipeline builder."""
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE, SCALED_FEATURES
from .features import ChurnFeatures


def _l1_logistic():
    """L1 logistic regression across scikit-learn versions (1.8 deprecates `penalty`)."""
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    kwargs = {"l1_ratio": 1.0} if (major, minor) >= (1, 8) else {"penalty": "l1"}
    return LogisticRegression(
        C=1.0, solver="saga", max_iter=5000, random_state=RANDOM_STATE, **kwargs
    )


def model_factories():
    """Candidate estimators, using the hyper-parameters from the notebook.

    Note: the notebook's "Logistic Regression" and "Ridge" models were the same
    estimator (scikit-learn's default penalty is L2), so they appear once here.
    """
    return {
        "Logistic Regression (L2)": lambda: LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE  # default penalty is L2
        ),
        "Logistic Regression (L1)": _l1_logistic,
        "Decision Tree": lambda: DecisionTreeClassifier(
            max_depth=5, min_samples_split=10, random_state=RANDOM_STATE
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    scaler = ColumnTransformer(
        [("scale", StandardScaler(), SCALED_FEATURES)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    scaler.set_output(transform="pandas")
    return Pipeline([
        ("features", ChurnFeatures()),
        ("scale", scaler),
        ("model", estimator),
    ])

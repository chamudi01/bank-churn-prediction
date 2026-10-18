"""Central configuration: paths, schema, modelling constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "Churn_Modelling.csv"
ARTIFACT_DIR = ROOT / "artifacts"

TARGET = "Exited"
ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]

# Raw inputs the model needs, in the order they are expected.
RAW_FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
NUMERIC_INPUTS = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
CATEGORICAL_INPUTS = {
    "Geography": ["France", "Germany", "Spain"],
    "Gender": ["Female", "Male"],
}

# Columns standardised after feature engineering (same set as the notebook).
SCALED_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "EstimatedSalary", "BalanceToSalaryRatio", "EngagementScore",
]

# Plausibility limits used when validating scoring input.
VALID_RANGES = {
    "CreditScore": (300, 900),
    "Age": (18, 100),
    "Tenure": (0, 50),
    "Balance": (0, None),
    "NumOfProducts": (1, 10),
    "EstimatedSalary": (0, None),
}
BINARY_INPUTS = ["HasCrCard", "IsActiveMember"]

# Risk bands on predicted churn probability. The lower edge sits at roughly the
# portfolio base rate (~20%); adjust to match your retention capacity.
RISK_BAND_EDGES = (0.20, 0.50)
RISK_BAND_LABELS = ("Low", "Medium", "High")

LABELS = {
    "CreditScore": "Credit score",
    "Geography": "Geography",
    "Gender": "Gender",
    "Age": "Age",
    "Tenure": "Tenure (years)",
    "Balance": "Account balance",
    "NumOfProducts": "Number of products",
    "HasCrCard": "Holds credit card",
    "IsActiveMember": "Active member",
    "EstimatedSalary": "Estimated salary",
}

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

"""Cached loaders and guard rails shared by every page."""
import streamlit as st

from churn.config import ARTIFACT_DIR, DEFAULT_DATA_PATH
from churn.data import clean, load_raw
from churn.predict import ArtifactsMissing, Bundle, load_bundle

TRAIN_COMMAND = "python -m churn.train"


@st.cache_resource(show_spinner=False)
def _bundle(mtime: float) -> Bundle:
    return load_bundle(ARTIFACT_DIR)


def get_bundle() -> Bundle | None:
    marker = ARTIFACT_DIR / "metadata.json"
    if not marker.exists():
        return None
    try:
        return _bundle(marker.stat().st_mtime)
    except ArtifactsMissing:
        return None


def require_bundle() -> Bundle:
    bundle = get_bundle()
    if bundle is None:
        st.error("Model artifacts were not found.")
        st.markdown(
            f"Train the models first from the project root:\n\n```\n{TRAIN_COMMAND}\n```\n"
            "This reads `data/raw/Churn_Modelling.csv` and writes the `artifacts/` folder."
        )
        st.stop()
    return bundle


@st.cache_data(show_spinner=False)
def _dataset(mtime: float):
    return clean(load_raw(DEFAULT_DATA_PATH))


def get_dataset():
    """Cleaned modelling dataset, or None when the CSV is not present."""
    if not DEFAULT_DATA_PATH.exists():
        return None
    return _dataset(DEFAULT_DATA_PATH.stat().st_mtime)


def active_model_name(bundle: Bundle) -> str:
    return st.session_state.get("model_name", bundle.metadata["selected_model"])

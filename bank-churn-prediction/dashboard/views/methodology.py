import streamlit as st

from churn.config import RISK_BAND_EDGES
from dashboard import components as ui
from dashboard import state

bundle = state.require_bundle()
meta = bundle.metadata

ui.page_header("Methodology", "How the models were built and how far to trust them.")

ui.section("Data")
st.markdown(
    f"- Source file: `{meta['data_file']}`, {meta['rows_raw']:,} customers.\n"
    f"- Identifier columns were removed, and {meta['rows_removed']:,} credit-score outliers "
    f"(beyond 1.5 × IQR) were excluded, leaving {meta['rows_modelled']:,} customers.\n"
    f"- Overall churn rate: {ui.pct(meta['churn_rate'])}. "
    f"Training split {meta['n_train']:,}, test split {meta['n_test']:,}, stratified on the outcome."
)

ui.section("Features")
st.markdown(
    "Ten customer attributes are transformed into 20 model inputs: balance-to-salary ratio, "
    "age band, tenure band, a balance flag, an engagement score "
    "(0.4 × products + 0.3 × active + 0.3 × card), a high-value flag, and one-hot geography. "
    "Medians behind the high-value flag and the scaling parameters are learned from the "
    "training split only."
)

ui.section("Models and selection")
st.markdown(
    "Three candidates are trained: L2 and L1 logistic regression, and a depth-5 decision tree. "
    f"The default model is chosen by {meta['selection_rule']}; the test split is used only for "
    "reporting. You can switch models from the sidebar."
)

ui.section("Risk bands")
lo, hi = RISK_BAND_EDGES
st.markdown(
    f"Low below {lo:.0%}, Medium from {lo:.0%} to {hi:.0%}, High at {hi:.0%} and above. "
    "These are operating choices, not model outputs; set them in `churn/config.py` to match "
    "retention capacity."
)

ui.section("Limitations")
st.markdown(
    "- Predictions describe statistical association in historical data, not causes.\n"
    "- The dataset is a single snapshot with no time dimension, so drift cannot be monitored "
    "and the models have not been validated out of time.\n"
    "- Recall at the default 0.50 threshold is modest; use the threshold control on the "
    "performance page to choose a trade-off.\n"
    "- Probabilities from a decision tree are coarse; treat them as ordering, not calibrated risk.\n"
    "- Gender and geography are model inputs. Review whether using them is appropriate for "
    "your jurisdiction and use case before acting on scores."
)

ui.section("Environment")
v = meta["versions"]
st.markdown(f"Trained {meta['trained_at']} with Python {v['python']}, scikit-learn "
            f"{v['scikit-learn']}, pandas {v['pandas']}. Retrain if you change library versions.")

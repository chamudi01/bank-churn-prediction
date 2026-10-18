import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from churn.config import ID_COLUMNS, RAW_FEATURES, RISK_BAND_LABELS
from churn.data import validate_scoring_frame
from churn.predict import risk_band, score
from dashboard import components as ui
from dashboard import state, theme

MAX_ROWS = 100_000

bundle = state.require_bundle()
selected = state.active_model_name(bundle)
pipe = bundle.models[selected]

ui.page_header(
    "Batch scoring",
    f"Score a customer file with the {selected} model and download the results.",
)

with st.expander("Required columns"):
    st.markdown(
        "The file must be a CSV containing: " + ", ".join(f"`{c}`" for c in RAW_FEATURES) + ". "
        "Other columns (such as `CustomerId`) are kept in the output. Geography must be France, "
        "Germany or Spain; Gender must be Female or Male; HasCrCard and IsActiveMember must be 0 or 1."
    )
    st.download_button("Download blank template", ",".join(RAW_FEATURES) + "\n",
                       file_name="scoring_template.csv", mime="text/csv")

upload = st.file_uploader("Customer file (CSV)", type=["csv"])
if upload is None:
    st.stop()

try:
    data = pd.read_csv(upload)
except Exception as exc:  # unreadable or not a CSV
    st.error(f"Could not read the file: {exc}")
    st.stop()

if len(data) > MAX_ROWS:
    st.error(f"The file has {len(data):,} rows; the limit is {MAX_ROWS:,}.")
    st.stop()
if data.empty:
    st.error("The file has no rows.")
    st.stop()

try:
    inputs = validate_scoring_frame(data)
except ValueError as exc:
    st.error("The file did not pass validation.")
    st.code(str(exc), language=None)
    st.stop()

result = data.copy()
result["churn_probability"] = score(pipe, inputs)
result["risk_band"] = risk_band(result["churn_probability"].to_numpy())
result = result.sort_values("churn_probability", ascending=False).reset_index(drop=True)

counts = result["risk_band"].value_counts().reindex(RISK_BAND_LABELS, fill_value=0)
k = st.columns(4)
k[0].metric("Customers scored", f"{len(result):,}")
for col, band in zip(k[1:], reversed(RISK_BAND_LABELS)):
    col.metric(f"{band} risk", f"{counts[band]:,}", f"{counts[band] / len(result):.1%} of file",
               delta_color="off")

ui.section("Distribution of scores")
fig = go.Figure(go.Histogram(x=result["churn_probability"], xbins=dict(start=0, end=1, size=0.05),
                             marker_color=theme.NAVY))
fig.update_xaxes(title="Predicted churn probability", tickformat=".0%", range=[0, 1])
fig.update_yaxes(title="Customers")
ui.chart(theme.style(fig, height=260))

ui.section("Scored customers", "Highest risk first.")
lead = [c for c in ID_COLUMNS if c in result.columns]
ordered = lead + ["churn_probability", "risk_band"] + [c for c in result.columns
                                                      if c not in lead + ["churn_probability", "risk_band"]]
st.dataframe(
    result[ordered], hide_index=True,
    column_config={"churn_probability": st.column_config.ProgressColumn(
        "Churn probability", min_value=0.0, max_value=1.0, format="%.3f")},
)
buffer = io.StringIO()
result[ordered].to_csv(buffer, index=False)
st.download_button("Download scored file", buffer.getvalue(),
                   file_name="churn_scores.csv", mime="text/csv")

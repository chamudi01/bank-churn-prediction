import plotly.graph_objects as go
import streamlit as st

from churn.config import LABELS
from churn.predict import baseline_row, response_curve
from dashboard import components as ui
from dashboard import state, theme

bundle = state.require_bundle()
selected = state.active_model_name(bundle)
pipe = bundle.models[selected]

ui.page_header(
    "Churn drivers",
    f"What the {selected} model relies on. Attributes are the original customer fields, "
    "not the engineered features.",
)

# --- permutation importance ----------------------------------------------------------
ui.section("Attribute importance",
           "Drop in test-set ROC-AUC when an attribute is shuffled, averaged over 10 repeats. "
           "Larger means the model depends on it more; it does not show direction.")
imp = (bundle.importance[bundle.importance["model"] == selected]
       .sort_values("importance_mean"))
fig = go.Figure(go.Bar(
    x=imp["importance_mean"], y=imp["feature"].map(LABELS), orientation="h",
    error_x=dict(type="data", array=imp["importance_std"], color=theme.SLATE, thickness=1),
    marker_color=theme.NAVY,
    hovertemplate="%{y}: %{x:.4f} AUC drop<extra></extra>",
))
fig.update_xaxes(title="Decrease in ROC-AUC", showgrid=True, gridcolor=theme.GRID)
fig.update_yaxes(showgrid=False)
ui.chart(theme.style(fig, height=380))

# --- response curve -----------------------------------------------------------------------
ui.section("Response curve",
           "Predicted churn probability for a reference customer as one attribute varies and "
           "all others stay fixed. Direction and shape of the relationship, as the model sees it.")
feature = st.selectbox("Attribute", list(LABELS), format_func=LABELS.get,
                       index=list(LABELS).index("Age"))
curve = response_curve(pipe, bundle.reference, feature)

fig = go.Figure()
if feature in bundle.reference["categorical"] or curve["value"].nunique() <= 5:
    fig.add_trace(go.Bar(x=curve["value"].astype(str), y=curve["probability"],
                         marker_color=theme.NAVY,
                         hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.update_xaxes(type="category")
else:
    fig.add_trace(go.Scatter(x=curve["value"], y=curve["probability"], mode="lines",
                             line=dict(color=theme.NAVY, width=2.5),
                             hovertemplate=f"{LABELS[feature]} %{{x:,.0f}}: %{{y:.1%}}<extra></extra>"))
fig.update_xaxes(title=LABELS[feature])
fig.update_yaxes(title="Predicted churn probability", tickformat=".0%", rangemode="tozero")
ui.chart(theme.style(fig, height=320))

base = baseline_row(bundle.reference)
shown = ", ".join(f"{LABELS[k].lower()} {v:,.0f}" if isinstance(v, float) else f"{LABELS[k].lower()} {v}"
                  for k, v in base.items() if k != feature)
ui.note(f"Reference customer: {shown}.")

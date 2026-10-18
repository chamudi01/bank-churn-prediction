import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from churn.evaluate import confusion_at, metrics_at, threshold_sweep
from dashboard import components as ui
from dashboard import state, theme

bundle = state.require_bundle()
meta = bundle.metadata
selected = state.active_model_name(bundle)
y = bundle.predictions["y_true"].to_numpy()

ui.page_header(
    "Model performance",
    f"Evaluated on {meta['n_test']:,} held-out customers that were not used for fitting. "
    f"Training used {meta['n_train']:,} customers.",
)

# --- comparison table ------------------------------------------------------------
ui.section("Model comparison", "Test-set metrics at a 0.50 decision threshold. "
           "Cross-validated ROC-AUC is measured on the training split only.")
rows = []
for name, m in meta["models"].items():
    t = m["test"]
    rows.append({
        "Model": name + ("  (default)" if name == meta["selected_model"] else ""),
        "CV ROC-AUC": m["cv_roc_auc_mean"], "Test ROC-AUC": t["roc_auc"],
        "Accuracy": t["accuracy"], "Precision": t["precision"],
        "Recall": t["recall"], "F1": t["f1"],
    })
table = pd.DataFrame(rows)
st.dataframe(
    table, hide_index=True,
    column_config={c: st.column_config.NumberColumn(c, format="%.3f") for c in table.columns[1:]},
)

# --- ROC -------------------------------------------------------------------------
ui.section("ROC curves")
palette = [theme.NAVY, theme.STEEL, "#C08A2E", theme.SLATE]
fig = go.Figure()
for i, name in enumerate(bundle.models):
    fpr, tpr, _ = roc_curve(y, bundle.predictions[name])
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=name,
        line=dict(color=palette[i % len(palette)], width=3 if name == selected else 1.5),
        opacity=1 if name == selected else 0.7,
    ))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                         line=dict(color=theme.BORDER, dash="dash", width=1)))
fig.update_xaxes(title="False positive rate", range=[0, 1])
fig.update_yaxes(title="True positive rate", range=[0, 1.01])
ui.chart(theme.style(fig, height=380, legend=True))

# --- threshold -------------------------------------------------------------------
ui.section(f"Decision threshold: {selected}",
           "Accuracy alone hides the trade-off in an imbalanced problem. Lowering the threshold "
           "catches more churners but flags more customers who would have stayed.")
prob = bundle.predictions[selected].to_numpy()
threshold = st.slider("Flag customers with predicted churn probability at or above",
                      0.05, 0.95, 0.50, 0.01)
m = metrics_at(y, prob, threshold)
cm = confusion_at(y, prob, threshold)
flagged = (cm["tp"] + cm["fp"]) / len(y)

k = st.columns(4)
k[0].metric("Recall", ui.pct(m["recall"]), help="Share of actual churners that are flagged.")
k[1].metric("Precision", ui.pct(m["precision"]), help="Share of flagged customers who actually churn.")
k[2].metric("F1", f"{m['f1']:.3f}")
k[3].metric("Customers flagged", ui.pct(flagged))

left, right = st.columns([1, 1.4])
with left:
    st.markdown("**Confusion matrix**")
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    heat = go.Figure(go.Heatmap(
        z=z, x=["Predicted stay", "Predicted churn"], y=["Actual stay", "Actual churn"],
        text=[[f"{v:,}" for v in r] for r in z], texttemplate="%{text}",
        colorscale=[[0, "#F3F5F8"], [1, theme.NAVY]], showscale=False,
        hovertemplate="%{y}, %{x}: %{z:,}<extra></extra>",
    ))
    heat.update_yaxes(autorange="reversed")
    ui.chart(theme.style(heat, height=300))
with right:
    st.markdown("**Precision and recall across thresholds**")
    sweep = pd.DataFrame(threshold_sweep(y, prob, np.linspace(0.05, 0.95, 91)))
    line = go.Figure()
    line.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["recall"], name="Recall",
                              line=dict(color=theme.NAVY, width=2)))
    line.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["precision"], name="Precision",
                              line=dict(color="#C08A2E", width=2)))
    line.add_vline(x=threshold, line_color=theme.SLATE, line_dash="dash", line_width=1)
    line.update_xaxes(title="Threshold")
    line.update_yaxes(tickformat=".0%", range=[0, 1.02])
    ui.chart(theme.style(line, height=300, legend=True))

if selected == "Decision Tree":
    ui.note("A depth-limited decision tree produces a small set of distinct probabilities, "
            "so metrics move in steps as the threshold changes.")

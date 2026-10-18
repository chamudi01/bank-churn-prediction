import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from churn.config import CATEGORICAL_INPUTS, TARGET
from dashboard import components as ui
from dashboard import state, theme

ui.page_header(
    "Portfolio overview",
    "Historical churn in the modelling population: the customers used to train and test the models.",
)

df = state.get_dataset()
if df is None:
    st.error("Dataset not found.")
    st.markdown("Place `Churn_Modelling.csv` in `data/raw/` and reload this page.")
    st.stop()

# --- filters -----------------------------------------------------------------
f1, f2, f3, f4 = st.columns([1.2, 1, 1.4, 1])
geo = f1.multiselect("Geography", CATEGORICAL_INPUTS["Geography"],
                     default=CATEGORICAL_INPUTS["Geography"])
gender = f2.multiselect("Gender", CATEGORICAL_INPUTS["Gender"],
                        default=CATEGORICAL_INPUTS["Gender"])
age_lo, age_hi = int(df["Age"].min()), int(df["Age"].max())
ages = f3.slider("Age", age_lo, age_hi, (age_lo, age_hi))
active_only = f4.selectbox("Membership", ["All", "Active", "Inactive"])

view = df[df["Geography"].isin(geo) & df["Gender"].isin(gender)
          & df["Age"].between(*ages)]
if active_only != "All":
    view = view[view["IsActiveMember"] == (1 if active_only == "Active" else 0)]

if view.empty:
    st.warning("No customers match the current filters.")
    st.stop()

overall = view[TARGET].mean()

# --- headline figures ----------------------------------------------------------
k = st.columns(5)
k[0].metric("Customers", f"{len(view):,}")
k[1].metric("Churn rate", ui.pct(overall))
k[2].metric("Churned customers", f"{int(view[TARGET].sum()):,}")
k[3].metric("Average balance", f"{view['Balance'].mean():,.0f}")
k[4].metric("Active members", ui.pct(view["IsActiveMember"].mean()))
if len(view) < len(df):
    ui.note(f"Filtered view. Unfiltered churn rate is {ui.pct(df[TARGET].mean())} "
            f"across {len(df):,} customers.")


def rate_by(group: pd.Series, order=None) -> pd.DataFrame:
    frame = pd.DataFrame({"group": group.astype(str), TARGET: view[TARGET].to_numpy()})
    out = frame.groupby("group", observed=True)[TARGET].agg(rate="mean", customers="count").reset_index()
    if order is not None:
        out["group"] = pd.Categorical(out["group"], categories=order, ordered=True)
        out = out.sort_values("group")
    return out


def rate_chart(data: pd.DataFrame, x_title: str):
    fig = px.bar(data, x="group", y="rate", custom_data=["customers"])
    fig.update_traces(
        marker_color=theme.NAVY,
        texttemplate="%{y:.1%}", textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>Churn rate %{y:.1%}<br>%{customdata[0]:,} customers<extra></extra>",
    )
    fig.add_hline(y=overall, line_dash="dash", line_color=theme.SLATE, line_width=1)
    fig.update_yaxes(tickformat=".0%", title="Churn rate",
                     range=[0, max(data["rate"].max() * 1.25, 0.05)])
    fig.update_xaxes(title=x_title, type="category")
    return theme.style(fig, height=300)


age_band = pd.cut(view["Age"], [17, 29, 39, 49, 59, 200],
                  labels=["18–29", "30–39", "40–49", "50–59", "60+"])
balance_band = pd.cut(view["Balance"], [-0.01, 0, 50_000, 100_000, 150_000, np.inf],
                      labels=["No balance", "Up to 50k", "50k–100k", "100k–150k", "Above 150k"])

ui.section("Where churn concentrates",
           "Dashed line marks the churn rate of the current selection. Hover for customer counts; "
           "small groups produce volatile rates.")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**By geography**")
    ui.chart(rate_chart(rate_by(view["Geography"]), "Geography"))
with c2:
    st.markdown("**By age band**")
    ui.chart(rate_chart(rate_by(age_band, list(age_band.cat.categories)), "Age band"))

c3, c4 = st.columns(2)
with c3:
    st.markdown("**By number of products**")
    ui.chart(rate_chart(rate_by(view["NumOfProducts"]), "Products held"))
with c4:
    st.markdown("**By account balance**")
    ui.chart(rate_chart(rate_by(balance_band, list(balance_band.cat.categories)), "Balance band"))

ui.section("Membership and gender")
c5, c6 = st.columns(2)
with c5:
    labels = view["IsActiveMember"].map({1: "Active", 0: "Inactive"})
    ui.chart(rate_chart(rate_by(labels), "Membership"))
with c6:
    ui.chart(rate_chart(rate_by(view["Gender"]), "Gender"))

import plotly.graph_objects as go
import streamlit as st

from churn.config import CATEGORICAL_INPUTS, LABELS
from churn.predict import attribute_effects, baseline_row, risk_band
from dashboard import components as ui
from dashboard import state, theme

bundle = state.require_bundle()
selected = state.active_model_name(bundle)
pipe = bundle.models[selected]
ref = bundle.reference
base = baseline_row(ref)
num = ref["numeric"]

ui.page_header(
    "Single customer scoring",
    f"Enter a customer profile to estimate churn probability with the {selected} model.",
)

with st.form("customer"):
    a, b, c = st.columns(3)
    credit = a.number_input("Credit score", 300, 900, int(base["CreditScore"]), 1)
    geography = b.selectbox("Geography", CATEGORICAL_INPUTS["Geography"],
                            index=CATEGORICAL_INPUTS["Geography"].index(base["Geography"]))
    gender = c.selectbox("Gender", CATEGORICAL_INPUTS["Gender"],
                         index=CATEGORICAL_INPUTS["Gender"].index(base["Gender"]))

    d, e, f = st.columns(3)
    age = d.number_input("Age", 18, 100, int(base["Age"]), 1)
    tenure = e.number_input("Tenure (years)", 0, 50, int(base["Tenure"]), 1)
    products = f.number_input("Number of products", 1, 10, int(base["NumOfProducts"]), 1)

    g, h, i = st.columns(3)
    balance = g.number_input("Account balance", 0.0, value=float(round(base["Balance"], 2)),
                             step=1000.0, format="%.2f")
    salary = h.number_input("Estimated salary", 0.0, value=float(round(base["EstimatedSalary"], 2)),
                            step=1000.0, format="%.2f")
    with i:
        card = st.checkbox("Holds credit card", value=bool(base["HasCrCard"]))
        active = st.checkbox("Active member", value=bool(base["IsActiveMember"]))

    submitted = st.form_submit_button("Score customer", type="primary")

if submitted:
    st.session_state["customer_row"] = {
        "CreditScore": credit, "Geography": geography, "Gender": gender, "Age": age,
        "Tenure": tenure, "Balance": balance, "NumOfProducts": products,
        "HasCrCard": int(card), "IsActiveMember": int(active), "EstimatedSalary": salary,
    }

row = st.session_state.get("customer_row")
if row is None:
    ui.note("Submit the form to see the result.")
    st.stop()

prob, effects = attribute_effects(pipe, row, ref)
band = risk_band(prob)
base_rate = ref["churn_rate"]

st.write("")
left, right = st.columns([1, 2])
with left:
    st.metric("Predicted churn probability", ui.pct(prob))
    st.markdown(ui.band_badge(band), unsafe_allow_html=True)
    st.write("")
    ui.note(f"Portfolio churn rate in training data: {ui.pct(base_rate)}. "
            f"This customer's score is {prob / base_rate:.1f} times that rate.")
    ui.note("Risk bands: Low below 20%, Medium 20–50%, High 50% and above.")

with right:
    ui.section("What moves this score",
               "Change in predicted probability if each attribute were set to the typical value "
               "(median or most common) instead. Red raises risk; blue lowers it.")
    eff = effects.reindex(effects["effect"].abs().sort_values().index)
    colors = [theme.POSITIVE if v > 0 else theme.NEGATIVE for v in eff["effect"]]
    fig = go.Figure(go.Bar(
        x=eff["effect"] * 100, y=eff["feature"].map(LABELS), orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x:+.1f} percentage points<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=theme.SLATE, line_width=1)
    fig.update_xaxes(title="Percentage points", showgrid=True, gridcolor=theme.GRID, zeroline=False)
    fig.update_yaxes(showgrid=False)
    ui.chart(theme.style(fig, height=360))

ui.note("Effects are measured one attribute at a time and ignore interactions, so they will not "
        "necessarily add up to the gap between this score and a typical customer's.")

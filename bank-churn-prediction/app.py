"""Entry point:  streamlit run app.py"""
import streamlit as st

from dashboard import state, theme

st.set_page_config(page_title="Customer Churn Analytics", layout="wide")
theme.inject()

pages = {
    "Analysis": [
        st.Page("dashboard/views/overview.py", title="Portfolio overview", default=True),
        st.Page("dashboard/views/model_performance.py", title="Model performance"),
        st.Page("dashboard/views/drivers.py", title="Churn drivers"),
    ],
    "Scoring": [
        st.Page("dashboard/views/single_customer.py", title="Single customer"),
        st.Page("dashboard/views/batch_scoring.py", title="Batch scoring"),
    ],
    "Reference": [
        st.Page("dashboard/views/methodology.py", title="Methodology"),
    ],
}

navigation = st.navigation(pages)

with st.sidebar:
    st.markdown("<div class='brand'>Customer Churn Analytics</div>"
                "<div class='brand-sub'>Retail banking portfolio</div>",
                unsafe_allow_html=True)

bundle = state.get_bundle()
if bundle is not None:
    names = list(bundle.models)
    default = bundle.metadata["selected_model"]
    st.session_state.setdefault("model_name", default)
    with st.sidebar:
        st.selectbox("Model", names, key="model_name",
                     help="Applies to scoring, performance and driver pages.")

navigation.run()

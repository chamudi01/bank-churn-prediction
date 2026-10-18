"""Visual system: one palette, one set of chart defaults, a small CSS layer."""
import streamlit as st

NAVY = "#1F3A5F"
STEEL = "#5B7FA6"
SLATE = "#5B6B7F"
INK = "#1B2430"
BORDER = "#DDE2E9"
GRID = "#EEF1F5"
SURFACE = "#F3F5F8"
RISK = {"Low": "#7C93AD", "Medium": "#C08A2E", "High": "#A33B3B"}
POSITIVE = "#A33B3B"   # raises churn risk
NEGATIVE = "#5B7FA6"   # lowers churn risk
FONT = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

_CSS = f"""
<style>
.block-container {{ padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1240px; }}
h1 {{ font-size: 1.6rem; font-weight: 600; color: {NAVY}; letter-spacing: -0.01em; margin-bottom: .15rem; }}
h2, h3 {{ font-weight: 600; color: {INK}; }}
h3 {{ font-size: 1.05rem; margin-top: 1.6rem; }}
[data-testid="stMetric"] {{ background: #fff; border: 1px solid {BORDER}; border-radius: 4px; padding: 12px 16px; }}
[data-testid="stMetricLabel"] p {{ font-size: .8rem; color: {SLATE}; }}
[data-testid="stMetricValue"] {{ font-size: 1.55rem; font-weight: 600; color: {NAVY}; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
[data-testid="stSidebar"] .brand {{ font-weight: 600; color: {NAVY}; font-size: 1.05rem; margin-bottom: .1rem; }}
[data-testid="stSidebar"] .brand-sub {{ color: {SLATE}; font-size: .8rem; margin-bottom: 1rem; }}
div[data-testid="stForm"] {{ border: 1px solid {BORDER}; border-radius: 4px; background: #fff; }}
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{ border-radius: 4px; }}
.band {{ display: inline-block; padding: 2px 10px; border-radius: 3px; font-size: .85rem; font-weight: 600; color: #fff; }}
.note {{ color: {SLATE}; font-size: .85rem; }}
footer, #MainMenu {{ visibility: hidden; }}
</style>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def style(fig, height: int = 320, legend: bool = False):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=4, r=4, t=8, b=4),
        font=dict(family=FONT, size=12, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickcolor=BORDER, title_font_size=12)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title_font_size=12)
    return fig

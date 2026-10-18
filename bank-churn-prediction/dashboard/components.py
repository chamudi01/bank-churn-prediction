"""Small reusable UI pieces."""
import streamlit as st

from . import theme


def page_header(title: str, description: str = "") -> None:
    st.title(title)
    if description:
        st.markdown(f"<div class='note'>{description}</div>", unsafe_allow_html=True)
    st.write("")


def section(title: str, note: str = "") -> None:
    st.subheader(title)
    if note:
        st.markdown(f"<div class='note'>{note}</div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div class='note'>{text}</div>", unsafe_allow_html=True)


def chart(fig) -> None:
    st.plotly_chart(fig, config={"displayModeBar": False})


def band_badge(band: str) -> str:
    return f"<span class='band' style='background:{theme.RISK[band]}'>{band} risk</span>"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"

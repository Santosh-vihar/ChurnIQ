"""
app.py — ChurnIQ Streamlit entry point

This file is intentionally minimal. It sets global page config and
defines the navigation between the two pages.
"""

import streamlit as st

# ── Page configuration (must be the very first Streamlit call) ───────────────
st.set_page_config(
    page_title="ChurnIQ — Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/combo-chart.png",
        width=64,
    )
    st.markdown("## ChurnIQ")
    st.markdown("*Customer Churn Intelligence*")
    st.divider()

    page = st.radio(
        "Navigate",
        options=["🏠  Home & Model Overview", "🔮  Predict & Upload"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Built with XGBoost + Streamlit")
    st.caption("© 2024 ChurnIQ")

# ── Route to pages ────────────────────────────────────────────────────────────
if page == "🏠  Home & Model Overview":
    from pages.home import render
    render()
else:
    from pages.predict import render
    render()

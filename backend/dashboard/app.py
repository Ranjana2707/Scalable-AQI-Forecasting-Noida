"""
dashboard/app.py
=================
Streamlit multi-page AQI forecasting dashboard.
Run: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Noida AQI Forecasting System",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("🌫️ Noida AQI Forecasting")
st.sidebar.markdown("**Multi-Station · 2015–2026**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 EDA Explorer",
    "🔮 AQI Forecast",
    "🧠 SHAP Explainability",
    "📈 Model Comparison",
])

st.sidebar.markdown("---")
st.sidebar.info(
    "**Stations:**\n"
    "- Sector-1 (UPPCB)\n"
    "- Sector-62 (CAAQMS)\n\n"
    "**Best model:** Hist Gradient Boosting\n"
    "RMSE = 1.53 | R² = 0.9998"
)

# ── Route to pages ────────────────────────────────────────────────────
if page == "🏠 Overview":
    from dashboard.pages.overview import render
    render()
elif page == "📊 EDA Explorer":
    from dashboard.pages.data_explorer import render
    render()
elif page == "🔮 AQI Forecast":
    from dashboard.pages.forecast import render
    render()
elif page == "🧠 SHAP Explainability":
    from dashboard.pages.explainability import render
    render()
elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison")
    import pandas as pd
    try:
        df = pd.read_csv("outputs/reports/models/complete_model_evaluation.csv")
        st.dataframe(df.sort_values("Test_RMSE"), use_container_width=True)
    except Exception:
        st.info("Run Phase 5 training to populate this table.")

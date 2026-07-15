"""
app.py — Production Streamlit entry point
==========================================
Run:  streamlit run app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Noida AQI Forecasting & XAI",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fa; }
    /* Sidebar branding */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2332 0%, #2c3e50 100%);
        color: white;
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stSelectbox label { color: #bdc3c7 !important; }
    /* Metric cards */
    [data-testid="stMetricValue"] { font-weight: 800; font-size: 1.6rem; }
    /* Tabs */
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.9rem; }
    /* Dataframe */
    .stDataFrame { border-radius: 8px; }
    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:16px 0 8px 0;">
        <div style="font-size:2.5rem;">🌫️</div>
        <div style="font-size:1.1rem; font-weight:800; letter-spacing:0.5px;">
            Noida AQI System
        </div>
        <div style="font-size:0.72rem; opacity:0.7; margin-top:4px;">
            Multi-Station · 2015–2026 · XAI
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        options=[
            "🏠  Overview",
            "📊  EDA Explorer",
            "🔮  Live Forecast",
            "🧠  SHAP Explainability",
            "📈  Model Comparison",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; opacity:0.75; line-height:1.7;">
        <b>Stations:</b><br>
        • Sector-1 (UPPCB)<br>
        • Sector-62 (CAAQMS)<br><br>
        <b>Best Model:</b><br>
        Hist Gradient Boosting<br>
        RMSE = 1.527 &nbsp;|&nbsp; R² = 0.9998<br><br>
        <b>Explainability:</b><br>
        Kernel SHAP · 1,085 samples<br>
        Base value = 297.5 AQI
    </div>
    """, unsafe_allow_html=True)

# ── Route to pages ─────────────────────────────────────────────────────
if "Overview" in page:
    from dashboard.pages.p1_overview import render
    render()

elif "EDA" in page:
    from dashboard.pages.p2_eda import render
    render()

elif "Forecast" in page:
    from dashboard.pages.p3_forecast import render
    render()

elif "SHAP" in page or "Explainability" in page:
    from dashboard.pages.p4_explainability import render
    render()

elif "Model" in page or "Comparison" in page:
    from dashboard.pages.p5_model_comparison import render
    render()

"""Page 1 — Overview."""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.data_loader import (
    load_master_data, load_ml_evaluation, get_aqi_category, STATION_LABELS
)
from dashboard.components.aqi_gauge import render_aqi_scale


def render():
    st.title("🌫️ Noida AQI Forecasting & Explainable AI System")
    st.markdown("**Multi-Station · 2015–2026 · Hist Gradient Boosting · SHAP Explainability**")

    render_aqi_scale()
    st.markdown("---")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best Model RMSE",  "1.527 AQI",  "↓ 94.8% vs baseline")
    c2.metric("Best Model R²",    "0.9998",     "Near-perfect fit")
    c3.metric("Dataset Size",     "8,391 rows", "2 stations · 11 yrs")
    c4.metric("Engineered Features", "59",      "Phase 4 pipeline")
    c5.metric("SHAP Samples",     "1,085",      "Full test set")

    st.markdown("---")
    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.subheader("📊 Dataset Overview")
        try:
            df = load_master_data()
            df_disp = df[["date","station","aqi","pm25","pm10","temperature"]].tail(20)
            st.dataframe(df_disp, use_container_width=True, height=280)
        except Exception as e:
            st.warning(f"Could not load data: {e}")

    with col2:
        st.subheader("🏆 Model Leaderboard")
        try:
            ev = load_ml_evaluation()
            test = ev[ev["Split"]=="Test"][["Model","Test_RMSE","Test_R2"]].sort_values("Test_RMSE").reset_index(drop=True)
            test.index += 1
            st.dataframe(test, use_container_width=True, height=280)
        except Exception as e:
            st.warning(f"Could not load evaluation: {e}")

    st.markdown("---")
    st.subheader("🗺️ Project Phases")
    phases = [
        ("1","Project Setup","✅"),("2","Data Preprocessing","✅"),
        ("3","EDA","✅"),("4","Feature Engineering","✅"),
        ("5","ML Baselines","✅"),("6","Deep Learning","✅"),
        ("7","SHAP XAI","✅"),("8","Dashboard","✅"),
    ]
    cols = st.columns(8)
    for col, (num, name, status) in zip(cols, phases):
        col.markdown(f"""
        <div style="text-align:center;background:#f0f4f8;border-radius:8px;padding:10px 4px;">
            <div style="font-size:1.4rem;">{status}</div>
            <div style="font-size:0.7rem;font-weight:700;color:#2c3e50;">Phase {num}</div>
            <div style="font-size:0.65rem;color:#666;">{name}</div>
        </div>""", unsafe_allow_html=True)

"""Page 2 — EDA Explorer."""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.data_loader import load_master_data, STATION_LABELS, MONTH_LABELS
from dashboard.utils.chart_utils import (
    timeseries_chart, annual_bar_chart, monthly_heatmap, seasonal_violin
)


def render():
    st.title("📊 EDA Explorer")
    st.markdown("Explore 11 years of historical AQI data across both Noida monitoring stations.")

    df = load_master_data()

    # Sidebar controls
    st.sidebar.markdown("### EDA Controls")
    station_opt = st.sidebar.selectbox("Station", ["Both","Sector-1 (UPPCB)","Sector-62 (CAAQMS)"])
    year_range  = st.sidebar.slider("Year range", 2015, 2026, (2015, 2026))
    roll_days   = st.sidebar.slider("Rolling window (days)", 7, 90, 30)

    df_f = df[(df["year"]>=year_range[0]) & (df["year"]<=year_range[1])]
    if station_opt != "Both":
        sid = "noida_sector_1" if "1" in station_opt else "noida_sector_62"
        df_f = df_f[df_f["station"]==sid]

    # Metrics
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Records",   f"{len(df_f):,}")
    c2.metric("Mean AQI",  f"{df_f['aqi'].mean():.1f}")
    c3.metric("Max AQI",   f"{df_f['aqi'].max():.0f}")
    c4.metric("Min AQI",   f"{df_f['aqi'].min():.0f}")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trend","📅 Annual","🌡️ Heatmap","🌸 Seasonal"])

    with tab1:
        st.pyplot(timeseries_chart(df_f, station_opt, roll_days), use_container_width=True)

    with tab2:
        st.pyplot(annual_bar_chart(df_f), use_container_width=True)

    with tab3:
        sid_heat = "noida_sector_62"
        if station_opt != "Both" and "1" in station_opt:
            sid_heat = "noida_sector_1"
        st.pyplot(monthly_heatmap(df_f, sid_heat), use_container_width=True)

    with tab4:
        if "season" in df_f.columns:
            st.pyplot(seasonal_violin(df_f), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Raw Data")
    st.dataframe(df_f.tail(50), use_container_width=True)

    # Download
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data (CSV)", csv,
                       "noida_aqi_filtered.csv", "text/csv")

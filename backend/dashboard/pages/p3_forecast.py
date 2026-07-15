"""Page 3 — Live AQI Forecast with SHAP waterfall."""
import streamlit as st
import pandas as pd
import numpy as np
import io, csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.data_loader import (
    load_model, load_feature_cols, load_train_data, load_shap_data,
    build_input_vector, get_aqi_category, STATION_MAP, MONTH_LABELS
)
from dashboard.components.aqi_gauge import render_aqi_card, render_aqi_scale
from dashboard.utils.shap_utils import compute_shap_single, waterfall_plot


def render():
    st.title("🔮 Real-Time AQI Forecast")
    st.markdown(
        "Configure pollutant concentrations and meteorological conditions to get an "
        "instant AQI prediction with SHAP-based explanation."
    )

    render_aqi_scale()

    model      = load_model()
    feat_cols  = load_feature_cols()
    train_df   = load_train_data()
    train_mean = train_df[feat_cols].mean()

    st.sidebar.markdown("### 🏭 Station & Date")
    station_opt = st.sidebar.selectbox("Station", list(STATION_MAP.keys()))
    month       = st.sidebar.selectbox("Month", range(1,13),
                                        format_func=lambda m: MONTH_LABELS[m-1],
                                        index=0)
    station_id  = 0 if "1" in station_opt else 1

    st.sidebar.markdown("### 💨 Pollutants (µg/m³ or mg/m³)")
    pm25  = st.sidebar.slider("PM₂.₅",     0.0, 500.0, 80.0,  1.0)
    pm10  = st.sidebar.slider("PM₁₀",      0.0, 800.0, 120.0, 1.0)
    no2   = st.sidebar.slider("NO₂",       0.0, 200.0, 40.0,  1.0)
    co    = st.sidebar.slider("CO (mg/m³)", 0.0, 20.0,  1.5,   0.1)
    so2   = st.sidebar.slider("SO₂",       0.0, 100.0, 10.0,  1.0)
    o3    = st.sidebar.slider("O₃",        0.0, 200.0, 35.0,  1.0)

    st.sidebar.markdown("### 🌤️ Meteorology")
    temp  = st.sidebar.slider("Temperature (°C)", -5.0, 50.0, 22.0, 0.5)
    hum   = st.sidebar.slider("Humidity (%)",      0.0, 100.0,60.0, 1.0)
    wind  = st.sidebar.slider("Wind Speed (km/h)", 0.0, 60.0,  8.0, 0.5)
    pres  = st.sidebar.slider("Pressure (hPa)",  970.0,1040.0,1010.0,0.5)

    st.sidebar.markdown("### 📅 Lag Values")
    lag1  = st.sidebar.slider("AQI yesterday",   0.0, 500.0, 180.0, 1.0)
    lag7  = st.sidebar.slider("AQI 7 days ago",  0.0, 500.0, 175.0, 1.0)
    lag30 = st.sidebar.slider("AQI 30 days ago", 0.0, 500.0, 200.0, 1.0)

    overrides = {
        "pm25": pm25, "pm10": pm10, "no2": no2, "co": co,
        "so2": so2, "o3": o3, "temperature": temp, "humidity": hum,
        "wind_speed": wind, "pressure": pres,
        "lag_aqi_1": lag1, "lag_aqi_7": lag7, "lag_aqi_30": lag30,
        "lag_pm25_1": pm25*0.9, "lag_pm25_7": pm25*0.85,
        "roll_mean_aqi_7": (lag1+lag7)/2,
        "roll_mean_pm25_7": pm25*0.88,
        "interact_co_pm25": co*pm25,
        "interact_pm25_humidity": pm25*hum/100,
        "station_encoded": float(station_id),
        "month": month,
        "aqi_yesterday": lag1,
    }

    X_input    = build_input_vector(feat_cols, train_mean, overrides)
    prediction = float(model.predict(X_input)[0])
    prediction = max(0, min(500, prediction))

    col_main, col_info = st.columns([1.3, 1])

    with col_main:
        render_aqi_card(prediction, station_opt)

        st.markdown("#### 🔑 SHAP Explanation")
        shap_data = load_shap_data()
        if shap_data is not None:
            X_bg = shap_data["X"][:150]
            with st.spinner("Computing SHAP values…"):
                shap_row, base_val = compute_shap_single(
                    model, X_bg, X_input[0], feat_cols)
            fig = waterfall_plot(shap_row, feat_cols, base_val, prediction)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("SHAP data not found. Run Phase 7 to enable explanations.")

    with col_info:
        st.markdown("#### 📋 Input Summary")
        inputs_display = pd.DataFrame({
            "Variable": ["PM₂.₅","PM₁₀","NO₂","CO","SO₂","O₃",
                         "Temperature","Humidity","Wind Speed","Pressure",
                         "AQI t−1","AQI t−7","Month","Station"],
            "Value":    [f"{pm25:.1f} µg/m³",f"{pm10:.1f} µg/m³",
                         f"{no2:.1f} µg/m³",f"{co:.2f} mg/m³",
                         f"{so2:.1f} µg/m³",f"{o3:.1f} µg/m³",
                         f"{temp:.1f} °C",f"{hum:.1f} %",
                         f"{wind:.1f} km/h",f"{pres:.1f} hPa",
                         f"{lag1:.0f}",f"{lag7:.0f}",
                         MONTH_LABELS[month-1], station_opt],
        })
        st.dataframe(inputs_display, use_container_width=True, height=350)

        # Download prediction
        cat, _, _ = get_aqi_category(prediction)
        report_df = pd.DataFrame([{
            "station": station_opt, "month": MONTH_LABELS[month-1],
            "predicted_aqi": round(prediction, 2), "aqi_category": cat,
            **{k: v for k,v in overrides.items() if k in feat_cols}
        }])
        csv_bytes = report_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Prediction Report (CSV)",
                           csv_bytes, "aqi_prediction.csv", "text/csv")

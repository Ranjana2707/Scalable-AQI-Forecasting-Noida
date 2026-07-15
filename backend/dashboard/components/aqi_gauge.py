"""dashboard/components/aqi_gauge.py — AQI gauge and category card."""
import streamlit as st
from dashboard.utils.data_loader import get_aqi_category, AQI_CATEGORIES


def render_aqi_card(predicted_aqi: float, station_label: str = "") -> None:
    """Render a styled AQI result card with colour, category and health advice."""
    cat, color, emoji = get_aqi_category(predicted_aqi)

    health_advice = {
        "Good":         "Air quality is satisfactory. Ideal for all outdoor activities.",
        "Satisfactory": "Air quality is acceptable. Sensitive individuals should monitor.",
        "Moderate":     "May affect sensitive groups. Reduce prolonged outdoor exertion.",
        "Poor":         "Everyone may experience health effects. Avoid long outdoor activity.",
        "Very Poor":    "Health warnings. Avoid outdoor activity. Use N95 masks.",
        "Severe":       "Emergency conditions. Stay indoors. Air purifiers recommended.",
    }

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}22 0%, {color}44 100%);
        border-left: 6px solid {color};
        border-radius: 10px;
        padding: 20px 24px;
        margin: 8px 0 16px 0;
    ">
        <div style="display:flex; align-items:center; gap:16px;">
            <span style="font-size:3rem;">{emoji}</span>
            <div>
                <div style="font-size:0.85rem; color:#666; font-weight:500; text-transform:uppercase; letter-spacing:1px;">
                    Predicted AQI {f'— {station_label}' if station_label else ''}
                </div>
                <div style="font-size:3rem; font-weight:900; color:{color}; line-height:1.1;">
                    {predicted_aqi:.0f}
                </div>
                <div style="font-size:1.25rem; font-weight:700; color:{color};">
                    {cat}
                </div>
            </div>
        </div>
        <div style="margin-top:12px; font-size:0.9rem; color:#444;">
            💡 {health_advice.get(cat, '')}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_aqi_scale() -> None:
    """Render the CPCB AQI colour scale legend."""
    cols = st.columns(len(AQI_CATEGORIES))
    for col, (lo, hi, cat, color, emoji) in zip(cols, AQI_CATEGORIES):
        col.markdown(f"""
        <div style="
            background-color:{color}33;
            border-top:4px solid {color};
            border-radius:6px;
            padding:8px 4px;
            text-align:center;
        ">
            <div style="font-size:1.2rem;">{emoji}</div>
            <div style="font-size:0.75rem; font-weight:700; color:{color};">{cat}</div>
            <div style="font-size:0.7rem; color:#666;">{lo}–{hi}</div>
        </div>
        """, unsafe_allow_html=True)

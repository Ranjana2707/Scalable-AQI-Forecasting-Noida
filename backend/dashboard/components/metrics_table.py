"""dashboard/components/metrics_table.py — Styled evaluation metrics display."""
import streamlit as st
import pandas as pd


def render_metrics_row(rmse: float, mae: float, mape: float, r2: float,
                       model_name: str = "Hist Gradient Boosting") -> None:
    st.markdown(f"**{model_name}** — Test Set (2025–2026)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"{rmse:.3f}", help="Root Mean Square Error (AQI units)")
    c2.metric("MAE",  f"{mae:.3f}",  help="Mean Absolute Error (AQI units)")
    c3.metric("MAPE", f"{mape:.2f}%",help="Mean Absolute Percentage Error")
    c4.metric("R²",   f"{r2:.4f}",   help="Coefficient of determination")


def render_comparison_table(df: pd.DataFrame, highlight_col: str = "Test_RMSE") -> None:
    """Render a styled comparison table with best model highlighted."""
    if df.empty:
        st.info("No evaluation data available.")
        return

    # Style: highlight best row (lowest RMSE)
    best_idx = df[highlight_col].idxmin() if highlight_col in df.columns else None

    def highlight_best(row):
        if row.name == best_idx:
            return ["background-color: #d5f5e3; font-weight: bold"] * len(row)
        return [""] * len(row)

    display_cols = [c for c in ["Model","Test_RMSE","Test_MAE","Test_MAPE","Test_R2","Test_sMAPE"]
                    if c in df.columns]
    if not display_cols:
        st.dataframe(df, use_container_width=True)
        return

    styled = df[display_cols].sort_values("Test_RMSE" if "Test_RMSE" in df.columns else display_cols[0]) \
                             .style.apply(highlight_best, axis=1) \
                             .format({c: "{:.4f}" for c in display_cols if c != "Model"})
    st.dataframe(styled, use_container_width=True, height=320)

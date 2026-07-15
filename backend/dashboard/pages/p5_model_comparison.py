"""Page 5 — Model Comparison & Evaluation."""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.data_loader import (
    load_model, load_test_data, load_feature_cols,
    load_ml_evaluation, load_dl_evaluation,
)
from dashboard.utils.chart_utils import model_comparison_chart, forecast_vs_actual_chart
from dashboard.components.metrics_table import render_metrics_row, render_comparison_table


def _style(ax, fig):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_facecolor("white"); fig.set_facecolor("white")
    ax.grid(True, color="#e8e8e8", linewidth=0.5, linestyle="--")


def render():
    st.title("📈 Model Comparison & Evaluation")
    st.markdown("Complete evaluation of all trained models on the held-out test set (2025–2026).")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Leaderboard",
        "📉 Predicted vs Actual",
        "📊 Residuals",
        "📋 Full Tables",
    ])

    ml_ev = load_ml_evaluation()
    dl_ev = load_dl_evaluation()

    # ── Tab 1: Leaderboard ────────────────────────────────────────────
    with tab1:
        st.subheader("Best Model — Hist Gradient Boosting")
        best = ml_ev[ml_ev["Split"]=="Test"].sort_values("Test_RMSE").iloc[0] \
               if not ml_ev.empty and "Split" in ml_ev.columns else None
        if best is not None:
            render_metrics_row(
                float(best["Test_RMSE"]), float(best["Test_MAE"]),
                float(best["Test_MAPE"]), float(best["Test_R2"]),
                best["Model"]
            )

        st.markdown("---")
        st.subheader("ML vs Deep Learning Comparison")
        if not ml_ev.empty:
            st.pyplot(model_comparison_chart(ml_ev, dl_ev), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**ML Models**")
            if not ml_ev.empty:
                test_ml = ml_ev[ml_ev["Split"]=="Test"] if "Split" in ml_ev.columns else ml_ev
                render_comparison_table(test_ml)
        with col2:
            st.markdown("**Deep Learning Models**")
            if not dl_ev.empty:
                render_comparison_table(dl_ev)
            else:
                st.info("No DL evaluation data.")

    # ── Tab 2: Predicted vs Actual ────────────────────────────────────
    with tab2:
        st.subheader("Forecast vs Actual — Test Period (2025–2026)")
        try:
            model      = load_model()
            test_df    = load_test_data()
            feat_cols  = load_feature_cols()
            y_pred     = model.predict(test_df[feat_cols].values)
            station_f  = st.selectbox("Station", ["Both","Sector-1","Sector-62"], key="ts_station")
            st.pyplot(
                forecast_vs_actual_chart(test_df, y_pred, station_f),
                use_container_width=True
            )

            # Scatter plot
            fig, ax = plt.subplots(figsize=(7, 5))
            y_true = test_df["aqi"].values
            ax.scatter(y_true, y_pred, s=5, alpha=0.30, color="#2980b9")
            lo = min(y_true.min(), y_pred.min())
            hi = max(y_true.max(), y_pred.max())
            ax.plot([lo,hi],[lo,hi], "k--", lw=1.3, label="Perfect fit")
            r2 = float(1 - np.sum((y_true-y_pred)**2)/np.sum((y_true-y_true.mean())**2))
            ax.set_xlabel("Actual AQI"); ax.set_ylabel("Predicted AQI")
            ax.set_title(f"Predicted vs Actual  (R²={r2:.4f})", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9); _style(ax, fig); plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load forecast data: {e}")

    # ── Tab 3: Residuals ──────────────────────────────────────────────
    with tab3:
        st.subheader("Residual Analysis — Best Model")
        try:
            model      = load_model()
            test_df    = load_test_data()
            feat_cols  = load_feature_cols()
            y_pred     = model.predict(test_df[feat_cols].values)
            y_true     = test_df["aqi"].values
            resid      = y_true - y_pred

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # Residual vs predicted
            axes[0].scatter(y_pred, resid, s=4, alpha=0.25, color="#8e44ad")
            axes[0].axhline(0, color="#e74c3c", lw=1.2, ls="--")
            axes[0].axhline(+2*resid.std(), color="#e67e22", lw=0.8, ls=":")
            axes[0].axhline(-2*resid.std(), color="#e67e22", lw=0.8, ls=":", label="±2σ")
            axes[0].set_xlabel("Predicted AQI"); axes[0].set_ylabel("Residual")
            axes[0].set_title(f"Residuals  (bias={resid.mean():.2f}, σ={resid.std():.2f})")
            axes[0].legend(fontsize=8); _style(axes[0], fig)

            # Histogram
            axes[1].hist(resid, bins=55, color="#8e44ad", alpha=0.72, density=True)
            from scipy.stats import norm as scipy_norm
            x = np.linspace(resid.min(), resid.max(), 200)
            axes[1].plot(x, scipy_norm.pdf(x, resid.mean(), resid.std()), "k--", lw=1.5)
            axes[1].axvline(0, color="#e74c3c", lw=1, ls="--")
            axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Density")
            axes[1].set_title("Residual Distribution"); _style(axes[1], fig)

            # Monthly residuals
            test_df2 = test_df.copy(); test_df2["resid"] = resid
            test_df2["month"] = test_df2["date"].dt.month
            m_means = test_df2.groupby("month")["resid"].mean()
            months  = ["J","F","M","A","M","J","J","A","S","O","N","D"]
            axes[2].bar(range(12), [m_means.get(m, 0) for m in range(1,13)],
                        color="#8e44ad", alpha=0.72)
            axes[2].axhline(0, color="black", lw=0.9)
            axes[2].set_xticks(range(12)); axes[2].set_xticklabels(months, fontsize=8)
            axes[2].set_title("Mean Residual by Month"); _style(axes[2], fig)

            fig.suptitle("Residual Analysis — Hist Gradient Boosting", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute residuals: {e}")

    # ── Tab 4: Full Tables ────────────────────────────────────────────
    with tab4:
        st.subheader("Complete Evaluation Tables")
        if not ml_ev.empty:
            st.markdown("**ML Model Evaluation (all splits)**")
            st.dataframe(ml_ev, use_container_width=True)
            csv = ml_ev.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download ML Evaluation (CSV)",
                               csv, "ml_evaluation.csv", "text/csv")
        if not dl_ev.empty:
            st.markdown("**Deep Learning Evaluation**")
            st.dataframe(dl_ev, use_container_width=True)
            csv = dl_ev.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download DL Evaluation (CSV)",
                               csv, "dl_evaluation.csv", "text/csv")

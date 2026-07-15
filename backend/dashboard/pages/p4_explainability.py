"""Page 4 — SHAP Explainability."""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.data_loader import (
    load_feature_importance, load_stationwise_shap,
    load_seasonal_shap, load_pollutant_shap, load_shap_data,
)
from dashboard.utils.shap_utils import global_importance_fig


def _style(ax, fig):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    fig.set_facecolor("white")
    ax.grid(True, color="#e8e8e8", linewidth=0.5, linestyle="--")


def render():
    st.title("🧠 Explainable AI — SHAP Analysis")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) quantifies every feature's contribution "
        "to each prediction. Base value = **297.5 AQI** (expected model output)."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 Global Importance",
        "🏭 Station Comparison",
        "🌸 Seasonal Analysis",
        "💨 Pollutants & Meteo",
        "📊 Beeswarm",
    ])

    # ── Tab 1: Global importance ──────────────────────────────────────
    with tab1:
        st.subheader("Top 20 Features — Global SHAP Importance")
        try:
            fi = load_feature_importance()
            st.pyplot(global_importance_fig(fi, top_n=20), use_container_width=True)
            st.markdown("---")
            st.subheader("Full Feature Importance Table")
            display_cols = [c for c in ["rank","feature","category",
                                         "fused_score","mean_abs_shap","perm_importance"]
                            if c in fi.columns]
            st.dataframe(fi[display_cols].head(30), use_container_width=True, height=400)
            csv = fi.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Feature Importance (CSV)",
                               csv, "feature_importance.csv", "text/csv")
        except Exception as e:
            st.warning(f"Feature importance not available: {e}")
            _try_show_img("figS7_global_importance_bar.png")

    # ── Tab 2: Station comparison ─────────────────────────────────────
    with tab2:
        st.subheader("Station-wise SHAP Comparison — Sector-1 vs Sector-62")
        try:
            s_df = load_stationwise_shap()
            top15 = s_df.head(15)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Grouped bar
            ax = axes[0]
            y  = np.arange(len(top15))
            w  = 0.38
            if "Sector1_Mean_SHAP" in s_df.columns:
                ax.barh(y-w/2, top15["Sector1_Mean_SHAP"],  height=w,
                        color="#1a6faf", alpha=0.82, label="Sector-1")
                ax.barh(y+w/2, top15["Sector62_Mean_SHAP"], height=w,
                        color="#c0392b", alpha=0.82, label="Sector-62")
            ax.set_yticks(y); ax.set_yticklabels(top15["Feature"], fontsize=8)
            ax.set_xlabel("Mean |SHAP|")
            ax.set_title("Station-wise Feature Importance", fontsize=10)
            ax.legend(fontsize=8); _style(ax, fig)

            # Difference
            ax = axes[1]
            if "Diff_S1_minus_S62" in top15.columns:
                diff = top15["Diff_S1_minus_S62"].values
                colors = ["#1a6faf" if v > 0 else "#c0392b" for v in diff]
                ax.barh(y, diff, color=colors, alpha=0.82, height=0.65)
                ax.axvline(0, color="black", lw=0.9)
            ax.set_yticks(y); ax.set_yticklabels(top15["Feature"], fontsize=8)
            ax.set_xlabel("Importance Difference (S1 − S62)")
            ax.set_title("Importance Difference", fontsize=10)
            _style(ax, fig)

            fig.suptitle("Station-wise SHAP Comparison", fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

            st.dataframe(s_df.head(20), use_container_width=True)
            csv = s_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Station SHAP (CSV)",
                               csv, "stationwise_shap.csv", "text/csv")
        except Exception as e:
            st.warning(f"Station SHAP not available: {e}")
            _try_show_img("figS8_stationwise_shap.png")

    # ── Tab 3: Seasonal ───────────────────────────────────────────────
    with tab3:
        st.subheader("Seasonal SHAP Analysis")
        try:
            seas_df = load_seasonal_shap()
            top12 = seas_df.head(12)
            season_cols = [c for c in ["Winter","Spring","Summer","Monsoon","Post-Monsoon"]
                           if c in top12.columns]
            if season_cols:
                heat_data = top12[season_cols].set_index(
                    top12["Feature"] if "Feature" in top12.columns else top12.index)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(heat_data, ax=ax, cmap="YlOrRd", annot=True, fmt=".1f",
                            annot_kws={"size":8}, linewidths=0.4,
                            cbar_kws={"label":"Mean |SHAP|","shrink":0.8})
                ax.set_title("Seasonal Importance Heatmap — Top 12 Features",
                             fontsize=11, fontweight="bold")
                ax.tick_params(labelsize=8)
                fig.set_facecolor("white")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            st.dataframe(seas_df.head(20), use_container_width=True)
            csv = seas_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Seasonal SHAP (CSV)",
                               csv, "seasonal_shap.csv", "text/csv")
        except Exception as e:
            st.warning(f"Seasonal SHAP not available: {e}")
            _try_show_img("figS9_seasonal_shap.png")

    # ── Tab 4: Pollutant & Meteo ──────────────────────────────────────
    with tab4:
        st.subheader("Pollutant and Meteorological SHAP Importance")
        col1, col2 = st.columns(2)
        try:
            p_df = load_pollutant_shap()
            with col1:
                st.markdown("**Pollutants ranked by mean |SHAP|**")
                st.dataframe(p_df, use_container_width=True, height=300)
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = plt.cm.Reds(np.linspace(0.4, 0.85, len(p_df)))
                ax.bar(range(len(p_df)), p_df["Mean_SHAP"], color=colors, alpha=0.85)
                ax.set_xticks(range(len(p_df)))
                ax.set_xticklabels(p_df["Pollutant"], rotation=30, ha="right", fontsize=9)
                ax.set_ylabel("Mean |SHAP|")
                ax.set_title("Pollutant SHAP Importance", fontsize=10)
                _style(ax, fig); plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
        except Exception as e:
            col1.warning(f"Pollutant SHAP not available: {e}")

        try:
            from dashboard.utils.data_loader import load_meteo_shap
            m_df = load_meteo_shap()
            with col2:
                st.markdown("**Meteorological variables ranked by mean |SHAP|**")
                st.dataframe(m_df, use_container_width=True, height=300)
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(m_df)))
                ax.bar(range(len(m_df)), m_df["Mean_SHAP"], color=colors, alpha=0.85)
                ax.set_xticks(range(len(m_df)))
                ax.set_xticklabels(m_df["Variable"], rotation=20, ha="right", fontsize=9)
                ax.set_ylabel("Mean |SHAP|")
                ax.set_title("Meteorological SHAP Importance", fontsize=10)
                _style(ax, fig); plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
        except Exception as e:
            col2.warning(f"Meteo SHAP not available: {e}")

    # ── Tab 5: Beeswarm ───────────────────────────────────────────────
    with tab5:
        st.subheader("SHAP Beeswarm — Distribution Across All Test Samples")
        shap_data = load_shap_data()
        if shap_data:
            shap_vals  = shap_data["shap_values"]
            feat_names = shap_data["feat_names"]
            fi         = load_feature_importance()
            top_n      = st.slider("Number of features to display", 10, 25, 15)
            top_feats  = fi["feature"].head(top_n).tolist()
            top_idx    = [feat_names.index(f) for f in top_feats if f in feat_names]
            sv_top     = shap_vals[:, top_idx]

            fig, ax = plt.subplots(figsize=(11, max(6, top_n*0.5)))
            for i, feat in enumerate(top_feats[:len(top_idx)]):
                sv_col = sv_top[:, i]
                fv_col = shap_data["X"][:, feat_names.index(feat)] if feat in feat_names else np.zeros(len(sv_col))
                vmin, vmax = np.percentile(fv_col, 5), np.percentile(fv_col, 95)
                norm = np.clip((fv_col - vmin) / max(vmax-vmin, 1e-9), 0, 1)
                rng = np.random.default_rng(i)
                jitter = rng.uniform(-0.35, 0.35, size=len(sv_col))
                ax.scatter(sv_col, i+jitter, c=norm, cmap="RdYlBu_r",
                           s=5, alpha=0.45, linewidths=0)

            ax.axvline(0, color="black", lw=0.9, ls="--")
            ax.set_yticks(range(len(top_feats[:len(top_idx)])))
            ax.set_yticklabels(top_feats[:len(top_idx)], fontsize=9)
            ax.set_xlabel("SHAP Value (impact on AQI prediction)", fontsize=10)
            ax.set_title("SHAP Beeswarm Summary Plot", fontsize=11, fontweight="bold")
            import matplotlib.colors as mcolors
            sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=mcolors.Normalize(0,1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
            cbar.set_label("Feature value (low→high)", fontsize=8)
            cbar.set_ticks([0,1]); cbar.set_ticklabels(["Low","High"], fontsize=8)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.set_facecolor("white"); fig.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("SHAP data not loaded. Run Phase 7 to generate SHAP values.")
            _try_show_img("figS1_shap_beeswarm_summary.png")


def _try_show_img(filename: str):
    img_path = Path(__file__).parent.parent.parent/"outputs/figures/shap"/filename
    if img_path.exists():
        st.image(str(img_path), use_column_width=True)

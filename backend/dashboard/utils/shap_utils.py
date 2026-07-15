"""dashboard/utils/shap_utils.py — SHAP computation + plot helpers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.explainability.shap_explainer import KernelSHAPExplainer

_CACHE = {}

def get_explainer(model, X_bg, feat_names):
    key = id(model)
    if key not in _CACHE:
        _CACHE[key] = KernelSHAPExplainer(model, X_bg, feature_names=feat_names, seed=42)
    return _CACHE[key]


def compute_shap_single(model, X_bg, x_single, feat_names):
    exp = get_explainer(model, X_bg, feat_names)
    return exp.shap_values(x_single.reshape(1,-1), n_coalitions=128)[0], exp.base_value


def waterfall_plot(shap_row, feat_names, base_value, prediction, top_n=12):
    abs_vals  = np.abs(shap_row)
    top_idx   = np.argsort(abs_vals)[::-1][:top_n]
    other_sum = shap_row[np.argsort(abs_vals)[::-1][top_n:]].sum()

    feats = [feat_names[i] for i in top_idx]
    vals  = list(shap_row[top_idx])
    if abs(other_sum) > 0.01:
        feats.append("Other features")
        vals.append(other_sum)

    # Sort by abs value for clean display
    order = np.argsort(np.abs(vals))
    feats = [feats[i] for i in order]
    vals  = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(5, len(feats)*0.6 + 1.5)))
    running = base_value
    for i, (feat, val) in enumerate(zip(feats, vals)):
        color = "#c0392b" if val > 0 else "#2980b9"
        ax.barh(i, val, left=running, color=color, alpha=0.85, height=0.68,
                edgecolor="white", linewidth=0.4)
        txt_x = running + val/2
        ax.text(txt_x, i, f"{val:+.1f}", va="center", ha="center",
                fontsize=8, fontweight="bold",
                color="white" if abs(val)>8 else "#2c3e50")
        running += val

    ax.axvline(base_value, color="#7f8c8d", lw=1.2, ls=":", label=f"Base: {base_value:.1f}")
    ax.axvline(prediction, color="#2c3e50", lw=2.0, ls="--", label=f"Pred: {prediction:.1f}")
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([f[:30] for f in feats], fontsize=9)
    ax.set_xlabel("AQI Prediction", fontsize=10)
    ax.set_title("SHAP Waterfall — Feature Contributions to Prediction",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(handles=[
        mpatches.Patch(color="#c0392b", label="↑ Increases AQI"),
        mpatches.Patch(color="#2980b9", label="↓ Decreases AQI"),
        plt.Line2D([0],[0], color="#7f8c8d", lw=1.2, ls=":", label=f"Base: {base_value:.1f}"),
        plt.Line2D([0],[0], color="#2c3e50", lw=2.0, ls="--", label=f"Pred: {prediction:.1f}"),
    ], fontsize=8, loc="lower right", framealpha=0.9)
    fig.set_facecolor("white"); ax.set_facecolor("white")
    plt.tight_layout()
    return fig


def global_importance_fig(fi_df, top_n=20):
    top = fi_df.head(top_n).iloc[::-1]
    CAT = {"Raw Pollutant/Meteo":"#c0392b","Rolling":"#1a6faf","Lag":"#27ae60",
           "Interaction":"#e67e22","Cyclical":"#16a085","Season":"#9b59b6",
           "Event":"#e74c3c","Station":"#7f8c8d"}
    colors = [CAT.get(c,"#34495e") for c in top["category"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top)), top["fused_score"].values, color=colors, alpha=0.85, height=0.72)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].tolist(), fontsize=9)
    ax.set_xlabel("Fused Importance Score", fontsize=9)
    ax.set_title("Global Feature Importance — Top 20 Features", fontsize=11, fontweight="bold")
    handles = [mpatches.Patch(color=c, label=k) for k,c in CAT.items()]
    ax.legend(handles=handles, fontsize=7.5, loc="lower right", framealpha=0.9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.set_facecolor("white"); ax.set_facecolor("white")
    plt.tight_layout()
    return fig

"""dashboard/utils/chart_utils.py — Reusable matplotlib chart builders."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

STATION_COLORS = {"noida_sector_1":"#1a6faf","noida_sector_62":"#c0392b"}
STATION_LABELS = {"noida_sector_1":"Sector-1 (UPPCB)","noida_sector_62":"Sector-62 (CAAQMS)"}
AQI_BAND_COLORS = [
    (0,50,"#009966"),(50,100,"#59BD45"),(100,200,"#FF9900"),
    (200,300,"#FF0000"),(300,400,"#99004C"),(400,500,"#7E0023"),
]


def _style(ax, fig):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    fig.set_facecolor("white")
    ax.grid(True, color="#e8e8e8", linewidth=0.5, linestyle="--")


def timeseries_chart(df: pd.DataFrame, station_filter: str = "Both",
                     rolling_days: int = 30) -> plt.Figure:
    """Rolling AQI time-series for one or both stations."""
    fig, ax = plt.subplots(figsize=(13, 4))
    stations = list(STATION_COLORS.keys()) if station_filter == "Both" \
               else [s for s in STATION_COLORS if station_filter in s]

    for sid in stations:
        sub = df[df["station"] == sid].set_index("date")["aqi"]
        roll = sub.rolling(rolling_days, center=True).mean()
        c = STATION_COLORS[sid]
        ax.fill_between(sub.index, sub.values, alpha=0.10, color=c)
        ax.plot(sub.index, sub.values, lw=0.5, alpha=0.35, color=c)
        ax.plot(roll.index, roll.values, lw=1.8, color=c,
                label=f"{STATION_LABELS[sid]} ({rolling_days}-day mean)")

    ax.axhline(300, color="#cc0033", lw=0.8, ls="--", alpha=0.6, label="Poor threshold (300)")
    ax.set_ylabel("AQI", fontsize=10)
    ax.set_title(f"AQI Trend — {rolling_days}-Day Rolling Mean", fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=9)
    _style(ax, fig)
    plt.tight_layout()
    return fig


def annual_bar_chart(df: pd.DataFrame) -> plt.Figure:
    """Annual mean AQI grouped bar chart."""
    ann = df.groupby(["year","station"])["aqi"].mean().unstack()
    years = ann.index.tolist()
    x = np.arange(len(years)); w = 0.38
    fig, ax = plt.subplots(figsize=(13, 4))
    for offset, (sid, label) in zip([-w/2, w/2], STATION_LABELS.items()):
        if sid in ann.columns:
            ax.bar(x+offset, ann[sid].values, width=w, color=STATION_COLORS[sid],
                   alpha=0.82, label=label)
    ax.axvspan(x[years.index(2020)] if 2020 in years else 0,
               x[years.index(2021)] if 2021 in years else 0,
               alpha=0.10, color="#e67e22", label="COVID-19")
    ax.set_xticks(x); ax.set_xticklabels(years, rotation=45, fontsize=8)
    ax.set_ylabel("Mean AQI"); ax.set_title("Annual Mean AQI by Station", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    _style(ax, fig); plt.tight_layout()
    return fig


def monthly_heatmap(df: pd.DataFrame, station_id: str) -> plt.Figure:
    """Year × Month AQI heatmap for one station."""
    sub = df[df["station"] == station_id]
    pivot = sub.pivot_table(values="aqi", index="year", columns="month", aggfunc="mean")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn_r", vmin=50, vmax=450,
                linewidths=0.3, linecolor="white",
                cbar_kws={"label":"Mean AQI","shrink":0.8},
                annot=True, fmt=".0f", annot_kws={"size":7})
    ax.set_title(f"Year × Month AQI Heatmap — {STATION_LABELS.get(station_id,station_id)}",
                 fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", labelsize=8)
    fig.set_facecolor("white"); plt.tight_layout()
    return fig


def seasonal_violin(df: pd.DataFrame) -> plt.Figure:
    """Seasonal AQI violin comparison between stations."""
    season_order = ["Winter","Spring","Summer","Monsoon","Post-Monsoon"]
    seasons_present = [s for s in season_order if s in df["season"].values]
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, season in enumerate(seasons_present):
        for offset, sid in [(-0.2,"noida_sector_1"),(+0.2,"noida_sector_62")]:
            data = df[(df["season"]==season)&(df["station"]==sid)]["aqi"].dropna().tolist()
            if len(data) < 5: continue
            vp = ax.violinplot([data], positions=[i+offset], widths=0.3,
                               showmedians=True, showextrema=False)
            for pc in vp["bodies"]:
                pc.set_facecolor(STATION_COLORS[sid]); pc.set_alpha(0.65)
            vp["cmedians"].set_color("white")
    ax.set_xticks(range(len(seasons_present)))
    ax.set_xticklabels(seasons_present, fontsize=9)
    ax.set_ylabel("AQI")
    ax.set_title("Seasonal AQI Distribution by Station", fontsize=11, fontweight="bold")
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(color=STATION_COLORS[s],label=STATION_LABELS[s])
                       for s in STATION_COLORS], fontsize=9)
    _style(ax, fig); plt.tight_layout()
    return fig


def forecast_vs_actual_chart(test_df: pd.DataFrame, y_pred: np.ndarray,
                              station_filter: str = "Both") -> plt.Figure:
    """Predicted vs actual AQI time-series on the test set."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    stations = list(STATION_COLORS.keys()) if station_filter == "Both" \
               else [s for s in STATION_COLORS if station_filter in s]

    test_copy = test_df.copy()
    test_copy["predicted"] = y_pred

    for ax, sid in zip(axes, stations):
        sub = test_copy[test_copy["station"]==sid]
        if sub.empty: ax.set_visible(False); continue
        ax.plot(sub["date"], sub["aqi"],       lw=1.0, color="#2c3e50", alpha=0.7, label="Actual")
        ax.plot(sub["date"], sub["predicted"], lw=1.2, color=STATION_COLORS[sid], alpha=0.9, label="Predicted")
        ax.fill_between(sub["date"], sub["aqi"], sub["predicted"], alpha=0.12, color=STATION_COLORS[sid])
        ax.set_ylabel("AQI"); ax.legend(fontsize=8)
        ax.set_title(f"{STATION_LABELS[sid]} — Forecast vs Actual (Test 2025–2026)", fontsize=10)
        _style(ax, fig)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].tick_params(axis="x", rotation=30)
    fig.suptitle("Model Forecast vs Actual AQI", fontsize=12, fontweight="bold")
    plt.tight_layout(); return fig


def model_comparison_chart(ml_df: pd.DataFrame, dl_df: pd.DataFrame) -> plt.Figure:
    """RMSE / R² grouped bar chart for all models."""
    frames = []
    if not ml_df.empty:
        sub = ml_df[ml_df["Split"]=="Test"][["Model","RMSE","R²"]].copy()
        sub["type"] = "ML"
        frames.append(sub)
    if not dl_df.empty:
        sub = dl_df[["Model","Test_RMSE","Test_R2"]].rename(
            columns={"Test_RMSE":"RMSE","Test_R2":"R²"})
        sub["type"] = "DL"
        frames.append(sub)
    if not frames:
        fig, ax = plt.subplots(); ax.text(0.5,0.5,"No data",ha="center"); return fig

    combined = pd.concat(frames, ignore_index=True).sort_values("RMSE")
    x = np.arange(len(combined)); w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#27ae60" if t=="ML" else "#8e44ad" for t in combined["type"]]
    axes[0].bar(x, combined["RMSE"], color=colors, alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(combined["Model"], rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("RMSE"); axes[0].set_title("Test RMSE (lower is better)", fontsize=10)

    axes[1].bar(x, combined["R²"], color=colors, alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(combined["Model"], rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("R²"); axes[1].set_title("Test R² (higher is better)", fontsize=10)
    axes[1].set_ylim(bottom=0.9)

    import matplotlib.patches as mp
    for ax in axes:
        ax.legend(handles=[mp.Patch(color="#27ae60",label="ML"),
                           mp.Patch(color="#8e44ad",label="DL")], fontsize=8)
        _style(ax, fig)

    fig.suptitle("Model Comparison — ML vs Deep Learning", fontsize=12, fontweight="bold")
    plt.tight_layout(); return fig

"""
src/features/rolling_features.py
==================================
Per-station rolling window statistics for AQI forecasting.

EDA justification
-----------------
From Fig 14d (rolling mean comparison), the 7-day and 30-day rolling
means track the AQI trajectory far better than the raw noisy signal.
Rolling statistics serve three purposes:
  1. Smoothed trend estimate (rolling mean, rolling median)
  2. Volatility / episode intensity (rolling std)
  3. Extreme event context (rolling max for pollution peaks)
  4. Directional momentum (rolling trend = slope of recent window)

Rolling windows chosen (days): [3, 7, 14, 30]
  - 3-day:  short-term smog episode detection
  - 7-day:  weekly cycle elimination + weekly trend
  - 14-day: bi-weekly slow pollution build-up
  - 30-day: monthly regime detection (monsoon onset, winter onset)

Leakage prevention
------------------
All rolling windows use ``min_periods=1`` so partial windows at the
start of each station's record don't produce unnecessary NaNs, BUT
``shift(1)`` is applied FIRST so that the rolling window ends at
time t-1, not t. This means the window for predicting AQI on day t
uses data up to and including day t-1 only.

Usage
-----
    from src.features.rolling_features import add_rolling_features
    df = add_rolling_features(df)
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_ROLLING_COLS: List[str] = [
    "aqi",    # primary target — most important rolling features
    "pm25",   # primary winter pollutant
    "pm10",   # primary summer pollutant
    "temperature",  # meteorological context
    "wind_speed",   # dispersion context
]

DEFAULT_ROLLING_WINDOWS: List[int] = [3, 7, 14, 30]

DEFAULT_ROLLING_STATS: List[str] = [
    "mean", "std", "min", "max", "median"
]


def add_rolling_features(
    df: pd.DataFrame,
    rolling_cols: Optional[List[str]] = None,
    windows: Optional[List[int]] = None,
    stats: Optional[List[str]] = None,
    include_trend: bool = True,
    trend_windows: Optional[List[int]] = None,
    station_col: str = "station",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Add rolling window statistics per station, shift-protected against leakage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame sorted by (station, date).
    rolling_cols : list of str, optional
        Columns to compute rolling stats for.
    windows : list of int, optional
        Rolling window sizes in days.
    stats : list of str, optional
        Statistics to compute: 'mean', 'std', 'min', 'max', 'median'.
    include_trend : bool
        If True, compute rolling linear slope (trend) for AQI.
    trend_windows : list of int, optional
        Windows for trend computation. Defaults to [7, 30].
    station_col : str
    date_col : str

    Returns
    -------
    pd.DataFrame
        With rolling feature columns appended.
        Naming: ``roll_<stat>_<col>_<window>``  (e.g. ``roll_mean_aqi_7``)
        Trend:  ``roll_trend_aqi_<window>``
    """
    rolling_cols  = rolling_cols  or DEFAULT_ROLLING_COLS
    windows       = windows       or DEFAULT_ROLLING_WINDOWS
    stats         = stats         or DEFAULT_ROLLING_STATS
    trend_windows = trend_windows or [7, 30]

    available_cols = [c for c in rolling_cols if c in df.columns]
    if len(available_cols) < len(rolling_cols):
        missing = [c for c in rolling_cols if c not in df.columns]
        logger.warning(f"Rolling columns not found (skipped): {missing}")

    logger.info(
        f"Computing rolling features | cols={available_cols} | "
        f"windows={windows} | stats={stats}"
    )

    df = df.copy().sort_values([station_col, date_col]).reset_index(drop=True)
    n_added = 0

    for col in available_cols:
        # Shift by 1 to avoid leakage: window ends at t-1 for predicting t
        shifted = df.groupby(station_col, observed=True)[col].shift(1)

        for w in windows:
            # Group by station for rolling (expanding min_periods=1 fills warm-up)
            base = shifted.groupby(df[station_col], observed=True)

            for stat in stats:
                feat_name = f"roll_{stat}_{col}_{w}"
                if stat == "mean":
                    df[feat_name] = (base.rolling(w, min_periods=1)
                                         .mean().reset_index(level=0, drop=True))
                elif stat == "std":
                    df[feat_name] = (base.rolling(w, min_periods=2)
                                         .std().reset_index(level=0, drop=True))
                elif stat == "min":
                    df[feat_name] = (base.rolling(w, min_periods=1)
                                         .min().reset_index(level=0, drop=True))
                elif stat == "max":
                    df[feat_name] = (base.rolling(w, min_periods=1)
                                         .max().reset_index(level=0, drop=True))
                elif stat == "median":
                    df[feat_name] = (base.rolling(w, min_periods=1)
                                         .median().reset_index(level=0, drop=True))
                n_added += 1

    # Rolling trend (linear slope over window)
    if include_trend and "aqi" in df.columns:
        for w in trend_windows:
            df[f"roll_trend_aqi_{w}"] = (
                df.groupby(station_col, observed=True)["aqi"]
                  .shift(1)
                  .groupby(df[station_col], observed=True)
                  .rolling(w, min_periods=max(3, w//2))
                  .apply(_linear_slope, raw=True)
                  .reset_index(level=0, drop=True)
            )
            n_added += 1

    # ── Derived composite rolling features ─────────────────────────────
    # AQI change rate: how fast is AQI rising/falling?
    if "roll_mean_aqi_3" in df.columns and "roll_mean_aqi_7" in df.columns:
        df["roll_aqi_momentum"] = (df["roll_mean_aqi_3"] - df["roll_mean_aqi_7"]).round(4)
        n_added += 1

    # Pollution episode flag: AQI 7-day rolling mean > 300 (Poor threshold)
    if "roll_mean_aqi_7" in df.columns:
        df["roll_episode_flag"] = (df["roll_mean_aqi_7"] > 300).astype("int8")
        n_added += 1

    logger.info(f"Rolling features added: {n_added}")
    return df


def _linear_slope(values: np.ndarray) -> float:
    """
    Compute the OLS slope of a 1D array against integer time indices.
    Returns the slope (AQI units per day) or NaN if insufficient data.
    """
    n = len(values)
    if n < 3:
        return np.nan
    x = np.arange(n, dtype=float)
    # Fast OLS slope
    xm = x - x.mean()
    ym = values - values.mean()
    denom = (xm * xm).sum()
    if denom == 0:
        return 0.0
    return float((xm * ym).sum() / denom)


def get_rolling_feature_names(
    rolling_cols: Optional[List[str]] = None,
    windows: Optional[List[int]] = None,
    stats: Optional[List[str]] = None,
    trend_windows: Optional[List[int]] = None,
) -> List[str]:
    """Return the complete list of rolling feature names this module produces."""
    rolling_cols  = rolling_cols  or DEFAULT_ROLLING_COLS
    windows       = windows       or DEFAULT_ROLLING_WINDOWS
    stats         = stats         or DEFAULT_ROLLING_STATS
    trend_windows = trend_windows or [7, 30]

    names = [
        f"roll_{stat}_{col}_{w}"
        for col in rolling_cols
        for w in windows
        for stat in stats
    ]
    names += [f"roll_trend_aqi_{w}" for w in trend_windows]
    names += ["roll_aqi_momentum", "roll_episode_flag"]
    return names

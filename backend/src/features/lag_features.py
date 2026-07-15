"""
src/features/lag_features.py
==============================
Per-station lag feature generation for AQI and all pollutants.

EDA justification
-----------------
From Fig 12 (autocorrelation analysis):
  ACF lag-1  = 0.9664  → yesterday's AQI explains 93% of variance
  ACF lag-7  = 0.9580  → strong weekly persistence
  ACF lag-14 = 0.9325  → bi-weekly memory
  ACF lag-30 = 0.8193  → monthly persistence still strong

This strong autocorrelation means lag features are the single most
powerful predictor family for this dataset. Every pollutant also
exhibits high temporal autocorrelation (PM2.5 r_lag1 ≈ 0.95).

Leakage prevention
------------------
Lags are computed PER STATION using groupby → shift. This guarantees
that station A's yesterday never appears as station B's lag, and that
the first row of any station correctly gets NaN (not another station's
last row).

After lag computation, NaN rows from the warm-up period are recorded
but NOT dropped here — the pipeline orchestrator decides whether to
drop or impute them to preserve data for sequence models.

Usage
-----
    from src.features.lag_features import add_lag_features
    df = add_lag_features(df, lag_cols=LAG_COLS, lag_days=[1,3,7,14,30])
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default configuration — driven by EDA findings
# ---------------------------------------------------------------------------

# Columns to lag (all key predictors identified in EDA)
DEFAULT_LAG_COLS: List[str] = [
    "aqi",    # ACF lag-1 = 0.97 — most important predictor
    "pm25",   # r=0.949 with AQI; lag-1 ≈ 0.95
    "pm10",   # r=0.928 with AQI
    "no2",    # r=0.745 with AQI
    "co",     # r=0.925 with AQI
    "o3",     # r=-0.360 — inverse (summer signal)
    "so2",    # r=0.180 (weak but used)
    "nh3",    # r=0.533 with AQI
    "temperature",  # r=-0.898 — strongest met predictor
    "wind_speed",   # r=-0.722 — dispersion proxy
    "humidity",     # r=0.428 with AQI
]

# Lag periods in days (daily data)
DEFAULT_LAG_DAYS: List[int] = [1, 2, 3, 7, 14, 30]


def add_lag_features(
    df: pd.DataFrame,
    lag_cols: Optional[List[str]] = None,
    lag_days: Optional[List[int]] = None,
    station_col: str = "station",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Add lag features for specified columns and lag periods.

    Each lag is computed per station using groupby → shift to prevent
    cross-station data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame sorted by (station, date). Must have ``date_col``
        and ``station_col`` columns.
    lag_cols : list of str, optional
        Columns to compute lags for. Defaults to ``DEFAULT_LAG_COLS``.
    lag_days : list of int, optional
        Lag periods in days. Defaults to ``DEFAULT_LAG_DAYS``.
    station_col : str
        Name of the station identifier column.
    date_col : str
        Name of the datetime column.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with lag columns appended.
        New column names: ``lag_<col>_<k>`` (e.g. ``lag_aqi_7``).

    Notes
    -----
    NaN values in the first ``max(lag_days)`` rows of each station are
    expected and intentional. Do NOT forward-fill them — they represent
    missing historical context.
    """
    lag_cols = lag_cols or DEFAULT_LAG_COLS
    lag_days = lag_days or DEFAULT_LAG_DAYS

    # Only lag columns that actually exist in the DataFrame
    available_cols = [c for c in lag_cols if c in df.columns]
    missing_cols   = [c for c in lag_cols if c not in df.columns]
    if missing_cols:
        logger.warning(f"Lag columns not found in DataFrame (skipped): {missing_cols}")

    logger.info(
        f"Computing lag features | cols={available_cols} | "
        f"lags={lag_days} | stations={df[station_col].unique().tolist()}"
    )

    df = df.copy().sort_values([station_col, date_col]).reset_index(drop=True)
    n_features_added = 0

    for col in available_cols:
        for k in lag_days:
            feat_name = f"lag_{col}_{k}"
            df[feat_name] = (
                df.groupby(station_col, observed=True)[col]
                  .shift(k)
            )
            n_features_added += 1

    logger.info(f"Lag features added: {n_features_added}")

    # Report warm-up NaN counts per station
    for sid in df[station_col].unique():
        mask = df[station_col] == sid
        max_lag = max(lag_days)
        nan_rows = df.loc[mask, f"lag_aqi_{max_lag}"].isna().sum()
        logger.debug(f"  [{sid}] {nan_rows} warm-up NaN rows (lag-{max_lag})")

    return df


def get_lag_feature_names(
    lag_cols: Optional[List[str]] = None,
    lag_days: Optional[List[int]] = None,
) -> List[str]:
    """Return the complete list of lag feature names this module produces."""
    lag_cols = lag_cols or DEFAULT_LAG_COLS
    lag_days = lag_days or DEFAULT_LAG_DAYS
    return [f"lag_{col}_{k}" for col in lag_cols for k in lag_days]


def compute_lag_correlation_table(
    df: pd.DataFrame,
    target_col: str = "aqi",
    lag_cols: Optional[List[str]] = None,
    lag_days: Optional[List[int]] = None,
    station_col: str = "station",
) -> pd.DataFrame:
    """
    Compute Pearson correlation between target and each lag feature,
    per station. Used for feature importance analysis in the research paper.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that already contains lag features (after add_lag_features).
    target_col : str
        Column to correlate against (usually 'aqi').
    lag_cols, lag_days : optional
        Same defaults as ``add_lag_features``.
    station_col : str

    Returns
    -------
    pd.DataFrame
        Columns: feature, station, pearson_r — sorted by |r| descending.
    """
    lag_cols = lag_cols or DEFAULT_LAG_COLS
    lag_days = lag_days or DEFAULT_LAG_DAYS
    rows = []
    for sid, grp in df.groupby(station_col, observed=True):
        for col in lag_cols:
            for k in lag_days:
                feat = f"lag_{col}_{k}"
                if feat not in grp.columns:
                    continue
                pair = grp[[target_col, feat]].dropna()
                if len(pair) < 10:
                    continue
                r = pair[target_col].corr(pair[feat])
                rows.append({"feature": feat, "station": sid,
                             "pearson_r": round(r, 4), "abs_r": round(abs(r), 4)})

    result = pd.DataFrame(rows).sort_values("abs_r", ascending=False)
    return result

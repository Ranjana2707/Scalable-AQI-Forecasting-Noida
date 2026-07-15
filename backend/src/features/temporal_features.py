"""
src/features/temporal_features.py
===================================
Temporal, calendar, cyclical, and seasonal feature generation.

Features produced
-----------------
Calendar
    year, month, day, day_of_week, day_of_year,
    week_of_year, quarter, is_weekend

Cyclical encodings (sin/cos pairs — avoids Dec→Jan discontinuity)
    sin_month, cos_month
    sin_day_of_year, cos_day_of_year
    sin_day_of_week, cos_day_of_week
    sin_week_of_year, cos_week_of_year

Season (one-hot encoded)
    season_Winter, season_Spring, season_Summer,
    season_Monsoon, season_Post_Monsoon

Special event flags
    is_covid_lockdown  (Mar 24 2020 – Jun 30 2021)
    aqi_saturated      (already in data; preserved here)
    is_diwali          (approximate: Oct 20 – Nov 5 each year)
    is_stubble_burning (Oct 1 – Nov 30 each year)

Holiday-ready architecture
    is_holiday column left as a stub (all zeros unless user provides
    a holiday calendar via the ``holiday_dates`` parameter).

EDA justification
-----------------
- Monthly pattern shows 5× AQI swing (summer 72 → winter 451).
- Cyclical encoding is mandatory: Dec→Jan must be continuous for LSTM.
- COVID flag captures the 38 % AQI drop structural break (Fig 11).
- Diwali flag captures annual Nov spike clearly visible in Fig 5b heatmap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEASON_MAP = {
    1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
    5: "Summer", 6: "Summer", 7: "Monsoon", 8: "Monsoon",
    9: "Monsoon", 10: "Post_Monsoon", 11: "Winter", 12: "Winter",
}

# Approximate Diwali date window (main festival ± 8 days)
# Diwali shifts by ~11 days each year; using Oct 20 – Nov 5 catches it every year
DIWALI_WINDOW = (10, 20, 11, 5)   # (start_month, start_day, end_month, end_day)

COVID_START = pd.Timestamp("2020-03-24")
COVID_END   = pd.Timestamp("2021-06-30")


def add_temporal_features(
    df: pd.DataFrame,
    date_col: str = "date",
    holiday_dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Add all temporal features to a DataFrame in-place (returns a copy).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a parsed datetime column.
    date_col : str
        Name of the datetime column.
    holiday_dates : list of str, optional
        ISO date strings for public holidays (e.g. ["2023-01-26"]).
        If None, ``is_holiday`` is added as all-zeros (architecture stub).

    Returns
    -------
    pd.DataFrame
        Original columns + temporal feature columns.

    Notes
    -----
    This function is applied to the full dataset BEFORE any train/test
    split. No statistics are learned here — all transformations are
    deterministic and therefore leak-free.
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col])

    logger.info(f"Adding temporal features to DataFrame shape={df.shape}")

    # ── 1. Calendar features ────────────────────────────────────────────
    df["year"]         = dt.dt.year.astype("int16")
    df["month"]        = dt.dt.month.astype("int8")
    df["day"]          = dt.dt.day.astype("int8")
    df["day_of_week"]  = dt.dt.dayofweek.astype("int8")      # 0=Mon, 6=Sun
    df["day_of_year"]  = dt.dt.dayofyear.astype("int16")
    df["week_of_year"] = dt.dt.isocalendar().week.astype("int8")
    df["quarter"]      = dt.dt.quarter.astype("int8")
    df["is_weekend"]   = (dt.dt.dayofweek >= 5).astype("int8")

    # ── 2. Cyclical encodings ────────────────────────────────────────────
    # Month (period = 12)
    df["sin_month"]        = np.sin(2 * np.pi * df["month"] / 12).round(6)
    df["cos_month"]        = np.cos(2 * np.pi * df["month"] / 12).round(6)
    # Day of year (period = 365.25)
    df["sin_day_of_year"]  = np.sin(2 * np.pi * df["day_of_year"] / 365.25).round(6)
    df["cos_day_of_year"]  = np.cos(2 * np.pi * df["day_of_year"] / 365.25).round(6)
    # Day of week (period = 7)
    df["sin_day_of_week"]  = np.sin(2 * np.pi * df["day_of_week"] / 7).round(6)
    df["cos_day_of_week"]  = np.cos(2 * np.pi * df["day_of_week"] / 7).round(6)
    # Week of year (period = 52)
    df["sin_week_of_year"] = np.sin(2 * np.pi * df["week_of_year"] / 52).round(6)
    df["cos_week_of_year"] = np.cos(2 * np.pi * df["week_of_year"] / 52).round(6)

    # ── 3. Season one-hot ────────────────────────────────────────────────
    df["season"] = df["month"].map(SEASON_MAP)
    season_dummies = pd.get_dummies(df["season"], prefix="season", dtype="int8")
    # Guarantee all 5 season columns even if some are absent in a split
    for s in ["Winter", "Spring", "Summer", "Monsoon", "Post_Monsoon"]:
        col = f"season_{s}"
        if col not in season_dummies.columns:
            season_dummies[col] = 0
    df = pd.concat([df, season_dummies[
        ["season_Winter","season_Spring","season_Summer",
         "season_Monsoon","season_Post_Monsoon"]
    ]], axis=1)

    # ── 4. Special event flags ───────────────────────────────────────────
    # COVID-19 national lockdown and subsequent restrictions
    df["is_covid_lockdown"] = (
        (dt >= COVID_START) & (dt <= COVID_END)
    ).astype("int8")

    # Diwali window: Oct 20 – Nov 5
    diwali_mask = (
        ((dt.dt.month == 10) & (dt.dt.day >= 20)) |
        ((dt.dt.month == 11) & (dt.dt.day <= 5))
    )
    df["is_diwali"] = diwali_mask.astype("int8")

    # Stubble burning season: Oct 1 – Nov 30
    stubble_mask = dt.dt.month.isin([10, 11])
    df["is_stubble_burning"] = stubble_mask.astype("int8")

    # Holiday stub (architecture-ready; populated by caller if available)
    if holiday_dates:
        holiday_ts = pd.to_datetime(holiday_dates)
        df["is_holiday"] = dt.dt.normalize().isin(holiday_ts).astype("int8")
    else:
        df["is_holiday"] = np.int8(0)

    # ── 5. AQI saturation flag (preserve or add) ─────────────────────────
    if "aqi_saturated" not in df.columns and "aqi" in df.columns:
        df["aqi_saturated"] = (df["aqi"] == 500).astype("int8")

    n_temporal = (
        8 + 8 + 5 + 4 + 1  # calendar + cyclical + season_ohe + events + holiday
    )
    logger.info(f"Temporal features added: ~{n_temporal} columns")
    return df


def get_temporal_feature_names() -> List[str]:
    """Return the complete list of feature names produced by this module."""
    return [
        # Calendar
        "year", "month", "day", "day_of_week", "day_of_year",
        "week_of_year", "quarter", "is_weekend",
        # Cyclical
        "sin_month", "cos_month",
        "sin_day_of_year", "cos_day_of_year",
        "sin_day_of_week", "cos_day_of_week",
        "sin_week_of_year", "cos_week_of_year",
        # Season OHE
        "season_Winter", "season_Spring", "season_Summer",
        "season_Monsoon", "season_Post_Monsoon",
        # Events
        "is_covid_lockdown", "is_diwali", "is_stubble_burning",
        "is_holiday", "aqi_saturated",
    ]

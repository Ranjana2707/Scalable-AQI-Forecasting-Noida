"""
src/features/pipeline.py
==========================
Orchestrates the complete Phase 4 feature engineering pipeline.

Pipeline execution order
------------------------
1.  Load master CSV (data/interim/noida_aqi_eda.csv)
2.  Add temporal features        (temporal_features.py)
3.  Add interaction features     (interaction_features.py)
4.  Add lag features             (lag_features.py)
5.  Add rolling features         (rolling_features.py)
6.  Apply StationEncoder         (station_features.py)  ← fit on train only
7.  Drop high-correlation / redundant features
8.  Apply temporal split labels
9.  Save full engineered dataset → data/processed/noida_features.csv
10. Save split CSVs              → data/processed/{train,val,test}_features.csv
11. Print feature inventory report

Leakage architecture
--------------------
The StationEncoder (step 6) is the only stateful transformer.
It is fit exclusively on the TRAINING split (date ≤ 2023-12-31)
and applied to all splits. All other transformers are stateless
(deterministic functions of the date and raw values) and are
therefore applied to the full dataset before splitting.

Usage
-----
    from src.features.pipeline import run_feature_pipeline
    result = run_feature_pipeline()

    # Access specific splits
    train_df = result["train"]
    val_df   = result["val"]
    test_df  = result["test"]
    feature_names = result["feature_names"]
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.temporal_features    import add_temporal_features, get_temporal_feature_names
from src.features.lag_features         import add_lag_features, get_lag_feature_names, compute_lag_correlation_table
from src.features.rolling_features     import add_rolling_features, get_rolling_feature_names
from src.features.interaction_features import add_interaction_features, get_interaction_feature_names
from src.features.station_features     import StationEncoder
from src.utils.logger                  import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Split boundaries (approved in project scope)
# ---------------------------------------------------------------------------
TRAIN_END = "2023-12-31"
VAL_END   = "2024-12-31"
# Test = 2025-01-01 → end of data

# ---------------------------------------------------------------------------
# Columns to EXCLUDE from the ML feature matrix (identifiers / targets / raw)
# ---------------------------------------------------------------------------
NON_FEATURE_COLS = {
    "date", "station", "aqi_category", "split", "season",
    # Raw pollutants are kept but their lags/rolling carry the predictive info.
    # We KEEP raw pollutants in case a model wants them as contemporaneous inputs.
}

# High-correlation drop list (correlation > 0.97 with another feature)
# Determined from EDA + feature correlation analysis
HIGH_CORR_DROP = [
    # year is nearly perfectly correlated with station_mean_aqi
    # (which is a better representation of the long-term level)
    # Keeping year for tree models; dropping for collinearity in linear
    # We will mark these but let the caller decide whether to drop
]


def run_feature_pipeline(
    input_path:  str = "data/interim/noida_aqi_eda.csv",
    output_dir:  str = "data/processed",
    lag_days:    Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    drop_warmup_rows: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Execute the full feature engineering pipeline end-to-end.

    Parameters
    ----------
    input_path : str
        Path to the EDA-ready master CSV.
    output_dir : str
        Directory to write all output CSVs.
    lag_days : list of int, optional
        Lag periods to generate. Defaults to [1, 2, 3, 7, 14, 30].
    rolling_windows : list of int, optional
        Rolling window sizes. Defaults to [3, 7, 14, 30].
    drop_warmup_rows : bool
        If True, drop rows with NaN lag features (warm-up period).
        The first max(lag_days) rows of each station become NaN.
    verbose : bool
        If True, print progress and summary.

    Returns
    -------
    dict with keys:
        "full"         : complete feature DataFrame
        "train"        : training split
        "val"          : validation split
        "test"         : test split
        "feature_names": list of ML-ready feature column names
        "station_encoder": fitted StationEncoder instance
        "lag_corr_table" : lag feature correlations with AQI
    """
    t0 = time.perf_counter()
    lag_days        = lag_days        or [1, 2, 3, 7, 14, 30]
    rolling_windows = rolling_windows or [3, 7, 14, 30]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _log = print if verbose else logger.debug

    _log("\n" + "="*65)
    _log("  PHASE 4 — FEATURE ENGINEERING PIPELINE")
    _log("="*65)

    # ── Step 1: Load ──────────────────────────────────────────────────
    _log("\n[1/9] Loading EDA dataset...")
    df = pd.read_csv(input_path, parse_dates=["date"])
    df = df.sort_values(["station", "date"]).reset_index(drop=True)
    _log(f"      Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ── Step 2: Temporal features ─────────────────────────────────────
    _log("[2/9] Adding temporal features...")
    df = add_temporal_features(df, date_col="date")
    _log(f"      Shape: {df.shape}")

    # ── Step 3: Interaction features ──────────────────────────────────
    _log("[3/9] Adding interaction features...")
    df = add_interaction_features(df)
    _log(f"      Shape: {df.shape}")

    # ── Step 4: Lag features ──────────────────────────────────────────
    _log(f"[4/9] Adding lag features (lags={lag_days})...")
    df = add_lag_features(df, lag_days=lag_days)
    _log(f"      Shape: {df.shape}")

    # ── Step 5: Rolling features ──────────────────────────────────────
    _log(f"[5/9] Adding rolling features (windows={rolling_windows})...")
    df = add_rolling_features(df, windows=rolling_windows)
    _log(f"      Shape: {df.shape}")

    # ── Step 6: Train/Val/Test split (needed for station encoder) ─────
    _log("[6/9] Splitting data (train≤2023, val=2024, test≥2025)...")
    df["split"] = "train"
    df.loc[df["date"].dt.year == 2024, "split"] = "val"
    df.loc[df["date"].dt.year >= 2025, "split"] = "test"

    split_counts = df.groupby(["station","split"]).size().unstack(fill_value=0)
    _log(f"\n{split_counts.to_string()}\n")

    train_mask = df["split"] == "train"
    val_mask   = df["split"] == "val"
    test_mask  = df["split"] == "test"

    # ── Step 7: StationEncoder (fit on train only) ────────────────────
    _log("[7/9] Fitting StationEncoder on train split...")
    encoder = StationEncoder()
    encoder.fit(df[train_mask])
    df = encoder.transform(df)
    _log(f"      Shape: {df.shape}")

    # ── Step 8: Drop warm-up NaN rows ────────────────────────────────
    max_lag = max(lag_days)
    if drop_warmup_rows:
        _log(f"[8/9] Dropping warm-up NaN rows (first {max_lag} per station)...")
        before = len(df)
        key_lag = f"lag_aqi_{max_lag}"
        if key_lag in df.columns:
            df = df.dropna(subset=[key_lag]).reset_index(drop=True)
        after = len(df)
        _log(f"      Dropped {before - after} warm-up rows | Remaining: {after:,}")
    else:
        _log("[8/9] Warm-up NaN rows retained (drop_warmup_rows=False).")

    # ── Step 9: Save outputs ──────────────────────────────────────────
    _log("[9/9] Saving feature datasets...")
    df.to_csv(output_path / "noida_features.csv", index=False)
    _log(f"      Full → {output_path}/noida_features.csv | shape={df.shape}")

    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        split_df = df[df["split"] == split_name].reset_index(drop=True)
        split_df.to_csv(output_path / f"{split_name}_features.csv", index=False)
        _log(f"      {split_name.capitalize()} → {len(split_df):,} rows")

    # ── Build feature name list ───────────────────────────────────────
    feature_names = _build_feature_list(df, verbose=verbose)

    # ── Lag correlation table ─────────────────────────────────────────
    lag_corr = compute_lag_correlation_table(df, target_col="aqi", lag_days=lag_days)
    lag_corr.to_csv(output_path / "lag_correlation_table.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    _log(f"\n{'='*65}")
    _log(f"  Feature engineering complete in {elapsed:.1f}s")
    _log(f"  Total features available  : {len(feature_names)}")
    _log(f"  Full dataset shape        : {df.shape}")
    _log(f"  Train rows                : {(df['split']=='train').sum():,}")
    _log(f"  Val rows                  : {(df['split']=='val').sum():,}")
    _log(f"  Test rows                 : {(df['split']=='test').sum():,}")
    _log(f"{'='*65}\n")

    return {
        "full":           df,
        "train":          df[df["split"] == "train"].reset_index(drop=True),
        "val":            df[df["split"] == "val"].reset_index(drop=True),
        "test":           df[df["split"] == "test"].reset_index(drop=True),
        "feature_names":  feature_names,
        "station_encoder":encoder,
        "lag_corr_table": lag_corr,
    }


def _build_feature_list(df: pd.DataFrame, verbose: bool = True) -> List[str]:
    """
    Identify all engineered feature columns (excludes target, identifiers).

    Returns
    -------
    list of str — sorted, deduplicated feature names.
    """
    exclude = {
        "date", "station", "aqi", "aqi_category", "split", "season",
    }
    raw_pollutants = {"pm25","pm10","no2","nh3","so2","co","o3","pb",
                      "temperature","humidity","wind_speed","wind_direction","pressure"}

    all_cols = set(df.columns)
    feature_cols = sorted(all_cols - exclude)

    # Categorise for reporting
    categories = {
        "raw_pollutants": [],
        "temporal_calendar": [],
        "temporal_cyclical": [],
        "temporal_season_ohe": [],
        "temporal_events": [],
        "lag": [],
        "rolling": [],
        "interaction": [],
        "station": [],
        "other": [],
    }

    for col in feature_cols:
        if col in raw_pollutants:
            categories["raw_pollutants"].append(col)
        elif col.startswith("lag_"):
            categories["lag"].append(col)
        elif col.startswith("roll_"):
            categories["rolling"].append(col)
        elif col.startswith("interact_"):
            categories["interaction"].append(col)
        elif col.startswith("station_"):
            categories["station"].append(col)
        elif col.startswith("sin_") or col.startswith("cos_"):
            categories["temporal_cyclical"].append(col)
        elif col.startswith("season_"):
            categories["temporal_season_ohe"].append(col)
        elif col in {"is_covid_lockdown","is_diwali","is_stubble_burning",
                     "is_holiday","aqi_saturated","is_weekend"}:
            categories["temporal_events"].append(col)
        elif col in {"year","month","day","day_of_week","day_of_year",
                     "week_of_year","quarter"}:
            categories["temporal_calendar"].append(col)
        else:
            categories["other"].append(col)

    if verbose:
        print("\n── FEATURE INVENTORY ─────────────────────────────────────")
        total = 0
        for cat, cols in categories.items():
            if cols:
                print(f"  {cat:<25} : {len(cols):>3} features")
                total += len(cols)
        print(f"  {'TOTAL':25} : {total:>3} features")
        print("─"*55)

    return feature_cols

"""
src/preprocessing/cleaner.py
==============================
Data cleaning pipeline: missing values, duplicates, type enforcement,
and temporal resampling for AQI time-series data.

Why each cleaning step matters for AQI forecasting
-----------------------------------------------------
- **Hourly resampling**: ML models require a regular time grid.  Gaps break
  lag features and recurrent networks silently.
- **Missing value imputation**: Time-aware interpolation preserves diurnal
  and weekly patterns better than mean/median filling.
- **Duplicate removal**: Duplicate timestamps distort rolling statistics and
  inflate training set size artificially.
- **Data type enforcement**: Numeric columns stored as strings cause
  ``NaN`` explosions during feature engineering.

Multi-station scalability
--------------------------
``DataCleaner`` is stateful (``fit`` learns statistics on training data,
``transform`` applies them).  A multi-station loop passes each station's
DataFrame separately — the same cleaner instance can be reused or a new
one instantiated per station depending on whether cross-station normalisation
is desired (currently: separate per station, future: shared).

Usage
-----
    from src.preprocessing.cleaner import DataCleaner

    cleaner = DataCleaner(config)
    df_clean = cleaner.fit_transform(df_validated)   # training data
    df_test_clean = cleaner.transform(df_test_raw)   # test / inference
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.preprocessing.validator import CANONICAL_SCHEMA
from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Numeric canonical columns (excluding date and station)
_NUMERIC_COLS: List[str] = [
    s.name for s in CANONICAL_SCHEMA if s.dtype == "float64"
]


class DataCleaner:
    """
    Stateful cleaning pipeline for validated AQI DataFrames.

    Fit on training data; transform on validation / test / inference data
    to prevent leakage of imputation statistics across splits.

    Parameters
    ----------
    config : AppConfig
        Project configuration driving all cleaning decisions.

    Attributes
    ----------
    column_medians : dict
        Per-column medians learned during ``fit_transform``; used as
        final-resort fallback when interpolation cannot fill edge NaNs.
    dropped_columns : list of str
        Columns dropped during ``fit_transform`` due to low coverage.
    is_fitted : bool
        True after ``fit_transform`` has been called at least once.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.date_col: str = "date"
        self.missing_strategy: str = config.preprocessing.missing_strategy
        self.interp_method: str = config.preprocessing.interpolation_method
        self.min_valid_pct: float = config.preprocessing.min_valid_data_pct

        # Learnt state
        self.column_medians: Dict[str, float] = {}
        self.dropped_columns: List[str] = []
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
        station_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fit cleaning statistics on ``df`` and return the cleaned copy.

        Call this on **training data only**.

        Parameters
        ----------
        df : pd.DataFrame
            Schema-validated DataFrame (output of ``SchemaValidator.validate``).
        station_id : str, optional
            Used for logging context.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with a ``DatetimeIndex`` and no missing
            numeric values.
        """
        sid = station_id or self.config.project.station_id
        logger.info(f"[{sid}] DataCleaner.fit_transform started | shape={df.shape}")
        df = df.copy()

        df = self._set_datetime_index(df, sid)
        df = self._remove_duplicate_timestamps(df, sid)
        df = self._resample_to_hourly(df, sid)
        df = self._drop_low_coverage_columns(df, sid, fit=True)
        df = self._learn_medians(df, sid)
        df = self._impute_missing(df, sid)
        df = self._enforce_non_negativity(df, sid)

        self.is_fitted = True
        logger.info(
            f"[{sid}] fit_transform complete | "
            f"shape={df.shape} | nulls={df.isna().sum().sum()}"
        )
        return df

    def transform(
        self,
        df: pd.DataFrame,
        station_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply learned cleaning parameters to new data (no refitting).

        Use this for validation / test sets and live inference.

        Parameters
        ----------
        df : pd.DataFrame
            Schema-validated DataFrame.
        station_id : str, optional
            Used for logging context.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame.

        Raises
        ------
        RuntimeError
            If called before ``fit_transform``.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Call fit_transform() on training data before transform()."
            )
        sid = station_id or self.config.project.station_id
        logger.info(f"[{sid}] DataCleaner.transform started | shape={df.shape}")
        df = df.copy()

        df = self._set_datetime_index(df, sid)
        df = self._remove_duplicate_timestamps(df, sid)
        df = self._resample_to_hourly(df, sid)

        # Drop same columns as during fit — do NOT relearn
        cols_to_drop = [c for c in self.dropped_columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.debug(f"[{sid}] Dropped (from fit): {cols_to_drop}")

        df = self._impute_missing(df, sid)
        df = self._enforce_non_negativity(df, sid)

        logger.info(f"[{sid}] transform complete | shape={df.shape}")
        return df

    # ------------------------------------------------------------------
    # Step 1 — Set DatetimeIndex
    # ------------------------------------------------------------------

    def _set_datetime_index(
        self, df: pd.DataFrame, station_id: str
    ) -> pd.DataFrame:
        """
        Set the ``date`` column as a ``DatetimeIndex`` and drop it as a column.

        Preserves the ``station`` column as a regular column (not the index)
        so that multi-station DataFrames can still be differentiated after
        concatenation.
        """
        if self.date_col not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                logger.debug(f"[{station_id}] DatetimeIndex already set.")
                return df
            raise ValueError(
                f"[{station_id}] Column '{self.date_col}' not found and index "
                "is not a DatetimeIndex.  Run SchemaValidator first."
            )

        df = df.set_index(self.date_col)
        df.index.name = "date"
        df.index = pd.to_datetime(df.index)
        logger.debug(f"[{station_id}] DatetimeIndex set.")
        return df

    # ------------------------------------------------------------------
    # Step 2 — Duplicate removal
    # ------------------------------------------------------------------

    def _remove_duplicate_timestamps(
        self, df: pd.DataFrame, station_id: str
    ) -> pd.DataFrame:
        """
        Remove rows with duplicate timestamps (keeping first occurrence).

        Why: Duplicate timestamps corrupt rolling-window features and
        cause off-by-one errors in sequence models.
        """
        before = len(df)
        # For multi-station DataFrames, duplicates are (timestamp, station) pairs
        if "station" in df.columns:
            duplicate_mask = df.index.duplicated(keep="first") & \
                             df["station"].duplicated(keep="first")
        else:
            duplicate_mask = df.index.duplicated(keep="first")

        df = df[~duplicate_mask]
        dropped = before - len(df)
        if dropped:
            logger.warning(
                f"[{station_id}] Removed {dropped:,} duplicate timestamps."
            )
        return df

    # ------------------------------------------------------------------
    # Step 3 — Hourly resampling
    # ------------------------------------------------------------------

    def _resample_to_hourly(
        self, df: pd.DataFrame, station_id: str
    ) -> pd.DataFrame:
        """
        Resample the time series to a regular 1-hour frequency.

        Sub-hourly readings are aggregated by mean.
        A gap of ≤ 3 hours is forward-filled immediately; larger gaps
        remain NaN for the interpolation stage.

        Why: A regular grid is mandatory for lag-based features and
        recurrent models (LSTM/GRU) which assume equal time steps.
        """
        original_len = len(df)

        # Separate station metadata before resampling (string col can't be meaned)
        station_col = df["station"].iloc[0] if "station" in df.columns else None
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_resampled = numeric_df.resample("1h").mean()

        # Short-gap forward fill (≤ 3 h)
        numeric_resampled = numeric_resampled.ffill(limit=3)

        # Re-attach station column
        if station_col is not None:
            numeric_resampled.insert(0, "station", station_col)

        new_len = len(numeric_resampled)
        logger.debug(
            f"[{station_id}] Resampled: {original_len} → {new_len} rows "
            f"({'+'if new_len >= original_len else ''}{new_len - original_len})"
        )
        return numeric_resampled

    # ------------------------------------------------------------------
    # Step 4 — Drop low-coverage columns (fit only)
    # ------------------------------------------------------------------

    def _drop_low_coverage_columns(
        self, df: pd.DataFrame, station_id: str, fit: bool
    ) -> pd.DataFrame:
        """
        Remove columns where fewer than ``min_valid_pct`` rows have values.

        These columns would create more noise than signal in models.
        The list of dropped columns is memorised so ``transform`` applies
        the same removal.
        """
        if not fit:
            return df

        to_drop: List[str] = []
        for col in df.select_dtypes(include=[np.number]).columns:
            valid_frac = df[col].notna().mean()
            if valid_frac < self.min_valid_pct:
                to_drop.append(col)
                logger.warning(
                    f"[{station_id}] Dropping '{col}': "
                    f"only {valid_frac:.1%} valid (threshold {self.min_valid_pct:.0%})."
                )

        self.dropped_columns = to_drop
        if to_drop:
            df = df.drop(columns=to_drop)
        logger.info(
            f"[{station_id}] Dropped {len(to_drop)} low-coverage column(s): {to_drop}"
        )
        return df

    # ------------------------------------------------------------------
    # Step 5 — Learn column medians (fit only)
    # ------------------------------------------------------------------

    def _learn_medians(
        self, df: pd.DataFrame, station_id: str
    ) -> pd.DataFrame:
        """
        Compute per-column medians on training data as a fallback imputation
        value for edge-of-series NaNs that interpolation cannot reach.

        Medians are used (not means) because AQI distributions are often
        right-skewed; mean would overestimate the typical value.
        """
        for col in df.select_dtypes(include=[np.number]).columns:
            median_val = df[col].median()
            self.column_medians[col] = float(median_val) if not np.isnan(median_val) else 0.0
        logger.debug(f"[{station_id}] Column medians learnt: {list(self.column_medians.keys())}")
        return df

    # ------------------------------------------------------------------
    # Step 6 — Impute missing values
    # ------------------------------------------------------------------

    def _impute_missing(
        self, df: pd.DataFrame, station_id: str
    ) -> pd.DataFrame:
        """
        Impute missing values using the configured strategy.

        Strategies
        ----------
        ``interpolate`` (default)
            Time-aware linear interpolation; forward-fill + backward-fill
            for edge NaNs; median fill as final safety net.
        ``ffill``
            Forward-fill up to 24 hours, then backward-fill.
        ``drop``
            Drop rows with any NaN in key pollutant columns.

        Why interpolation is preferred for AQI
        ----------------------------------------
        Air pollutant concentrations change smoothly over hours.  Linear
        interpolation between neighbouring observations produces physically
        realistic gap-fills that preserve diurnal cycles — essential for
        LSTM and rolling-window features.
        """
        missing_before = df.select_dtypes(include=[np.number]).isna().sum().sum()

        if self.missing_strategy == "interpolate":
            df = df.interpolate(method=self.interp_method, limit_direction="both")
            df = df.ffill().bfill()
            # Final fallback: median fill for any remaining NaN (e.g., all-NaN col)
            for col, median_val in self.column_medians.items():
                if col in df.columns and df[col].isna().any():
                    df[col] = df[col].fillna(median_val)

        elif self.missing_strategy == "ffill":
            df = df.ffill(limit=24).bfill()

        elif self.missing_strategy == "drop":
            key_cols = [c for c in ["aqi", "pm25", "pm10"] if c in df.columns]
            df = df.dropna(subset=key_cols)

        else:
            raise ValueError(
                f"Unknown missing_strategy '{self.missing_strategy}'. "
                "Choose: interpolate | ffill | drop"
            )

        missing_after = df.select_dtypes(include=[np.number]).isna().sum().sum()
        logger.info(
            f"[{station_id}] Missing values: {missing_before:,} → {missing_after:,} "
            f"(strategy='{self.missing_strategy}')"
        )
        return df

    # ------------------------------------------------------------------
    # Step 7 — Non-negativity enforcement
    # ------------------------------------------------------------------

    def _enforce_non_negativity(
        self, df: pd.DataFrame, station_id: str
    ) -> pd.DataFrame:
        """
        Clip physically non-negative columns to ≥ 0.

        Interpolation across a near-zero gap can produce tiny negatives
        (e.g., -0.001 for PM2.5).  These are physically impossible and
        would corrupt log-transforms used in some model architectures.
        """
        non_negative_cols = [
            "aqi", "pm25", "pm10", "no2", "so2", "co",
            "o3", "nh3", "humidity", "wind_speed",
        ]
        clipped = []
        for col in non_negative_cols:
            if col in df.columns:
                neg_count = int((df[col] < 0).sum())
                if neg_count > 0:
                    df[col] = df[col].clip(lower=0.0)
                    clipped.append(f"{col}({neg_count})")

        if clipped:
            logger.debug(
                f"[{station_id}] Clipped negative values in: {clipped}"
            )
        return df

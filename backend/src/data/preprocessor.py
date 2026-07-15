"""
src/data/preprocessor.py
==========================
Full preprocessing pipeline for AQI time-series data.

Pipeline stages (in order)
---------------------------
1. Set datetime index
2. Resample to hourly frequency (forward-fill small gaps)
3. Handle missing values (interpolation / ffill / model-based)
4. Detect and cap outliers (IQR or Z-score)
5. Drop low-coverage columns
6. Normalise / scale numerical features

Each stage is a separate method and can be called independently
for unit testing or partial pipeline runs.

Usage
-----
    from src.data.preprocessor import AQIPreprocessor
    from src.utils.config import load_config

    cfg = load_config("configs/default.yaml")
    preprocessor = AQIPreprocessor(cfg)
    df_clean = preprocessor.fit_transform(df_raw)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.utils.config import AppConfig
from src.utils.io_helpers import save_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AQIPreprocessor:
    """
    Stateful preprocessing pipeline for AQI time-series data.

    Designed to be ``fit`` on training data and ``transform`` on validation /
    test data to prevent data leakage.

    Parameters
    ----------
    config : AppConfig
        Project configuration driving all preprocessing decisions.

    Attributes
    ----------
    scaler : RobustScaler | None
        Fitted scaler (available after ``fit_transform`` is called).
    dropped_columns : list of str
        Columns removed due to low data coverage.
    outlier_bounds : dict
        Per-column (lower, upper) bounds used for outlier capping.
    is_fitted : bool
        True after ``fit_transform`` has been called at least once.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.dt_col: str = config.data.datetime_col
        self.target_col: str = config.data.target_col
        self.pollutant_cols: List[str] = config.data.pollutant_cols
        self.meteo_cols: List[str] = config.data.meteorological_cols

        # Preprocessing hyperparams from config
        self.missing_strategy: str = config.preprocessing.missing_strategy
        self.interp_method: str = config.preprocessing.interpolation_method
        self.outlier_method: str = config.preprocessing.outlier_method
        self.outlier_threshold: float = config.preprocessing.outlier_threshold
        self.min_valid_pct: float = config.preprocessing.min_valid_data_pct

        # State (set during fit)
        self.scaler: Optional[RobustScaler] = None
        self.scaler_cols: List[str] = []
        self.dropped_columns: List[str] = []
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the preprocessing pipeline on ``df`` and transform it.

        This is the method to call on **training data**.  Use ``transform``
        for validation / test data after this has been called.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame from the loader (datetime column present but not
            yet set as the index).

        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame with a DatetimeIndex.
        """
        logger.info("Starting fit_transform preprocessing pipeline.")
        df = df.copy()
        df = self._set_datetime_index(df)
        df = self._resample_hourly(df)
        df = self._drop_low_coverage_columns(df, fit=True)
        df = self._handle_missing_values(df)
        df = self._handle_outliers(df, fit=True)
        df = self._add_aqi_category(df)
        self.is_fitted = True
        logger.info(f"Preprocessing complete | final shape={df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using parameters learned during ``fit_transform``.

        Use this for validation / test sets and live inference.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame with the same schema as the training data.

        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame.

        Raises
        ------
        RuntimeError
            If called before ``fit_transform``.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Call fit_transform() on training data before transform()."
            )
        logger.info("Starting transform preprocessing pipeline.")
        df = df.copy()
        df = self._set_datetime_index(df)
        df = self._resample_hourly(df)

        # Drop columns identified during fit (do NOT refit)
        existing_dropped = [c for c in self.dropped_columns if c in df.columns]
        if existing_dropped:
            df = df.drop(columns=existing_dropped)
            logger.debug(f"Dropped columns (from fit): {existing_dropped}")

        df = self._handle_missing_values(df)
        df = self._handle_outliers(df, fit=False)  # apply learned bounds only
        df = self._add_aqi_category(df)
        logger.info(f"Transform complete | shape={df.shape}")
        return df

    # ------------------------------------------------------------------
    # Stage 1: Datetime index
    # ------------------------------------------------------------------

    def _set_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert the datetime column to a DatetimeIndex."""
        if self.dt_col not in df.columns:
            raise ValueError(
                f"Datetime column '{self.dt_col}' not found. "
                "Run DataLoader first."
            )
        df = df.set_index(self.dt_col)
        df.index = pd.to_datetime(df.index)
        df.index.name = "datetime"
        logger.debug("Datetime index set.")
        return df

    # ------------------------------------------------------------------
    # Stage 2: Hourly resampling
    # ------------------------------------------------------------------

    def _resample_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample to a regular 1-hour frequency.

        Sub-hourly data is aggregated by mean.
        Gaps up to 3 hours are forward-filled; longer gaps remain NaN
        for the imputation stage to handle.
        """
        original_len = len(df)
        df = df.resample("1h").mean()

        # Forward-fill tiny gaps (≤ 3 consecutive hours)
        df = df.fillna(method="ffill", limit=3)

        new_len = len(df)
        logger.debug(
            f"Resampled to hourly: {original_len} → {new_len} rows "
            f"({new_len - original_len:+d} rows)"
        )
        return df

    # ------------------------------------------------------------------
    # Stage 3: Drop low-coverage columns
    # ------------------------------------------------------------------

    def _drop_low_coverage_columns(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Remove columns with fewer than ``min_valid_pct`` non-null values."""
        if not fit:
            return df

        low_cov = []
        for col in df.columns:
            valid_frac = df[col].notna().mean()
            if valid_frac < self.min_valid_pct:
                low_cov.append(col)
                logger.warning(
                    f"Dropping '{col}' — only {valid_frac:.1%} valid values "
                    f"(threshold: {self.min_valid_pct:.0%})."
                )

        self.dropped_columns = low_cov
        if low_cov:
            df = df.drop(columns=low_cov)
        logger.info(f"Dropped {len(low_cov)} low-coverage column(s): {low_cov}")
        return df

    # ------------------------------------------------------------------
    # Stage 4: Missing-value imputation
    # ------------------------------------------------------------------

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using the configured strategy.

        Strategies
        ----------
        interpolate : Time-aware linear interpolation (default).
        ffill       : Forward-fill then backward-fill.
        drop        : Drop rows with any remaining NaN.
        """
        missing_before = df.isna().sum().sum()

        if self.missing_strategy == "interpolate":
            df = df.interpolate(method=self.interp_method, limit_direction="both")
            # Fallback for any remaining NaN at edges
            df = df.fillna(method="ffill").fillna(method="bfill")

        elif self.missing_strategy == "ffill":
            df = df.fillna(method="ffill").fillna(method="bfill")

        elif self.missing_strategy == "drop":
            df = df.dropna()

        else:
            raise ValueError(
                f"Unknown missing_strategy: '{self.missing_strategy}'. "
                "Choose from: interpolate | ffill | drop."
            )

        missing_after = df.isna().sum().sum()
        logger.info(
            f"Missing values: {missing_before:,} → {missing_after:,} "
            f"(strategy='{self.missing_strategy}')"
        )
        return df

    # ------------------------------------------------------------------
    # Stage 5: Outlier detection and capping
    # ------------------------------------------------------------------

    def _handle_outliers(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """
        Detect and cap outliers using IQR or Z-score method.

        Outliers are *capped* (Winsorized) rather than dropped to preserve
        the time-series continuity required for lag-based models.

        Parameters
        ----------
        df : pd.DataFrame
        fit : bool
            If True, compute bounds from ``df`` (training).
            If False, apply pre-computed bounds (inference).
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        total_capped = 0

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue

            if fit:
                if self.outlier_method == "iqr":
                    q1, q3 = series.quantile([0.25, 0.75])
                    iqr_val = q3 - q1
                    lo = q1 - self.outlier_threshold * iqr_val
                    hi = q3 + self.outlier_threshold * iqr_val
                elif self.outlier_method == "zscore":
                    mu, sigma = series.mean(), series.std()
                    lo = mu - self.outlier_threshold * sigma
                    hi = mu + self.outlier_threshold * sigma
                else:
                    raise ValueError(
                        f"Unknown outlier_method: '{self.outlier_method}'. "
                        "Choose: iqr | zscore."
                    )
                # Enforce physical non-negativity for concentration columns
                if col in self.pollutant_cols or col == self.target_col:
                    lo = max(lo, 0.0)
                self.outlier_bounds[col] = (lo, hi)
            else:
                if col not in self.outlier_bounds:
                    continue
                lo, hi = self.outlier_bounds[col]

            before = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lower=lo, upper=hi)
            total_capped += int(before)

        logger.info(
            f"Outlier capping complete | method='{self.outlier_method}' | "
            f"cells capped={total_capped:,}"
        )
        return df

    # ------------------------------------------------------------------
    # Stage 6: AQI category (ordinal label)
    # ------------------------------------------------------------------

    def _add_aqi_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add an ordinal ``aqi_category`` column based on CPCB breakpoints.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with a new ``aqi_category`` column (int 0–5).
        """
        if self.target_col not in df.columns:
            logger.warning(
                f"Target column '{self.target_col}' not found; "
                "skipping AQI category encoding."
            )
            return df

        bp = self.config.data.aqi_breakpoints
        bins = [
            bp["good"][0],
            bp["good"][1],
            bp["satisfactory"][1],
            bp["moderate"][1],
            bp["poor"][1],
            bp["very_poor"][1],
            bp["severe"][1] + 1,  # +1 so 500 falls in 'severe'
        ]
        labels = [0, 1, 2, 3, 4, 5]  # ordinal encoding
        label_names = ["good", "satisfactory", "moderate", "poor", "very_poor", "severe"]

        df["aqi_category"] = pd.cut(
            df[self.target_col],
            bins=bins,
            labels=labels,
            right=True,
            include_lowest=True,
        ).astype("Int64")

        df["aqi_category_name"] = pd.cut(
            df[self.target_col],
            bins=bins,
            labels=label_names,
            right=True,
            include_lowest=True,
        ).astype(str)

        logger.debug("AQI category columns added.")
        return df

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_processed(self, df: pd.DataFrame, output_dir: Optional[str] = None) -> Path:
        """
        Save the processed DataFrame to the configured outputs directory.

        Parameters
        ----------
        df : pd.DataFrame
        output_dir : str, optional
            Override the default processed data directory.

        Returns
        -------
        Path
            Path to the saved CSV file.
        """
        dest_dir = Path(output_dir or self.config.paths.processed_data)
        dest_path = dest_dir / f"{self.config.project.station_id}_processed.csv"
        save_dataframe(df, dest_path, index=True)
        logger.info(f"Processed data saved → {dest_path}")
        return dest_path

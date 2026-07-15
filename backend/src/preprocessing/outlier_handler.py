"""
src/preprocessing/outlier_handler.py
======================================
Outlier detection and treatment for AQI time-series data.

Methods available
-----------------
- **IQR**       : Interquartile Range fencing (robust, non-parametric).
- **Z-Score**   : Standard-deviation based (assumes normality; use with care).
- **Modified Z**: Median-based Z-score (robust alternative to Z-score).
- **Rolling IQR**: Per-window IQR for detecting local spikes.

Treatment strategies
--------------------
- **cap**   : Winsorize to (lower, upper) fence — default; preserves time continuity.
- **nan**   : Replace outliers with NaN for re-imputation.
- **drop**  : Remove outlier rows (use with extreme caution for time-series).

Why capping is preferred for AQI
----------------------------------
AQI data contains real episodic spikes (festivals, crop burning, dust storms).
Dropping or NaN-replacing these destroys genuine signal.  Capping at the fence
retains the temporal structure while preventing extreme values from dominating
gradient-based models.

Multi-station scalability
--------------------------
``OutlierHandler`` learns bounds from one station's training data.  For
multi-station deployment, instantiate one handler per station and serialise
each handler's ``bounds`` dictionary alongside its station model artifact.

Usage
-----
    from src.preprocessing.outlier_handler import OutlierHandler

    handler = OutlierHandler(config)
    df_clean = handler.fit_transform(df_cleaned)   # training
    df_test  = handler.transform(df_test)           # val / test
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.preprocessing.validator import CANONICAL_SCHEMA, SCHEMA_MAP
from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns where outlier treatment is applied
_TREATMENT_COLS: List[str] = [
    s.name for s in CANONICAL_SCHEMA
    if s.dtype == "float64" and s.name not in ("temperature",)
    # Temperature CAN be negative — excluded from non-negativity clipping
    # but still outlier-treated with its own physical bounds
]


class OutlierHandler:
    """
    Detects and treats outliers in numeric AQI columns.

    Parameters
    ----------
    config : AppConfig
        Project configuration.

    Attributes
    ----------
    bounds : dict
        Per-column ``{col: (lower_fence, upper_fence)}`` learned during fit.
    method : str
        Detection method: ``iqr`` | ``zscore`` | ``modified_zscore``.
    treatment : str
        Treatment strategy: ``cap`` | ``nan`` | ``drop``.
    threshold : float
        Multiplier for IQR or sigma; controls sensitivity.
    is_fitted : bool
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.method: str = config.preprocessing.outlier_method
        self.threshold: float = config.preprocessing.outlier_threshold
        self.treatment: str = "cap"   # Hardcoded as safest for time-series

        self.bounds: Dict[str, Tuple[float, float]] = {}
        self.outlier_counts: Dict[str, int] = {}
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
        Learn outlier bounds from ``df`` and apply treatment.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned DataFrame (output of ``DataCleaner``).
        station_id : str, optional
            Logging context.

        Returns
        -------
        pd.DataFrame
            DataFrame with outliers treated.
        """
        sid = station_id or self.config.project.station_id
        logger.info(
            f"[{sid}] OutlierHandler.fit_transform | method={self.method} | "
            f"threshold={self.threshold} | treatment={self.treatment}"
        )
        df = df.copy()
        cols = self._get_numeric_cols(df)

        for col in cols:
            series = df[col].dropna()
            if len(series) < 20:
                logger.debug(f"[{sid}] '{col}': too few values ({len(series)}) — skipping.")
                continue

            lo, hi = self._compute_bounds(series, col)
            self.bounds[col] = (lo, hi)
            df = self._apply_treatment(df, col, lo, hi, sid)

        self.is_fitted = True
        total_outliers = sum(self.outlier_counts.values())
        logger.info(
            f"[{sid}] Outlier treatment complete | "
            f"total cells treated={total_outliers:,} | "
            f"columns={list(self.outlier_counts.keys())}"
        )
        return df

    def transform(
        self,
        df: pd.DataFrame,
        station_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply learned bounds to new data (no refitting).

        Parameters
        ----------
        df : pd.DataFrame
        station_id : str, optional

        Returns
        -------
        pd.DataFrame

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
        df = df.copy()

        for col, (lo, hi) in self.bounds.items():
            if col in df.columns:
                df = self._apply_treatment(df, col, lo, hi, sid)

        logger.info(f"[{sid}] OutlierHandler.transform complete.")
        return df

    def get_outlier_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame summarising outlier counts and bounds per column.

        Returns
        -------
        pd.DataFrame
            Columns: col, lower_bound, upper_bound, outliers_treated.
        """
        rows = []
        for col, (lo, hi) in self.bounds.items():
            rows.append({
                "column": col,
                "lower_bound": round(lo, 3),
                "upper_bound": round(hi, 3),
                "outliers_treated": self.outlier_counts.get(col, 0),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Bound computation
    # ------------------------------------------------------------------

    def _compute_bounds(
        self, series: pd.Series, col: str
    ) -> Tuple[float, float]:
        """
        Compute (lower, upper) fences using the configured method.

        Physical bounds from ``SCHEMA_MAP`` are used as hard outer limits —
        the statistical fence cannot exceed the physical maximum of the sensor.
        """
        if self.method == "iqr":
            lo, hi = self._iqr_bounds(series)
        elif self.method == "zscore":
            lo, hi = self._zscore_bounds(series)
        elif self.method == "modified_zscore":
            lo, hi = self._modified_zscore_bounds(series)
        else:
            raise ValueError(
                f"Unknown outlier_method '{self.method}'. "
                "Choose: iqr | zscore | modified_zscore"
            )

        # Apply physical constraints from the schema
        spec = SCHEMA_MAP.get(col)
        if spec:
            if spec.physical_min is not None:
                lo = max(lo, spec.physical_min)
            if spec.physical_max is not None:
                hi = min(hi, spec.physical_max)

        return float(lo), float(hi)

    def _iqr_bounds(
        self, series: pd.Series
    ) -> Tuple[float, float]:
        """
        IQR fencing: Q1 - k*IQR, Q3 + k*IQR  (k = threshold, default 3.0).

        k=3.0 is more lenient than the classic k=1.5 to avoid flagging genuine
        pollution spikes as outliers.  Tuned for AQI data.
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return q1 - self.threshold * iqr, q3 + self.threshold * iqr

    def _zscore_bounds(
        self, series: pd.Series
    ) -> Tuple[float, float]:
        """
        Z-score fencing: mean ± k*std  (k = threshold).

        Use only when the column is approximately normally distributed.
        """
        mu = series.mean()
        sigma = series.std()
        return mu - self.threshold * sigma, mu + self.threshold * sigma

    def _modified_zscore_bounds(
        self, series: pd.Series
    ) -> Tuple[float, float]:
        """
        Modified Z-score: uses median and MAD instead of mean and std.
        More robust for skewed distributions like PM2.5.

        Reference: Iglewicz & Hoaglin (1993).  Threshold ≈ 3.5 recommended.
        """
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0:
            # Degenerate case: fall back to IQR
            return self._iqr_bounds(series)
        modified_z = 0.6745 * (series - median) / mad
        safe_threshold = self.threshold * 1.167  # scale to match Z-score semantics
        fence = safe_threshold * mad / 0.6745
        return float(median - fence), float(median + fence)

    # ------------------------------------------------------------------
    # Treatment application
    # ------------------------------------------------------------------

    def _apply_treatment(
        self,
        df: pd.DataFrame,
        col: str,
        lo: float,
        hi: float,
        station_id: str,
    ) -> pd.DataFrame:
        """Apply the configured treatment to outlier values in ``col``."""
        mask = (df[col] < lo) | (df[col] > hi)
        count = int(mask.sum())

        if count == 0:
            return df

        self.outlier_counts[col] = self.outlier_counts.get(col, 0) + count
        pct = count / len(df[col].dropna()) * 100

        if self.treatment == "cap":
            df[col] = df[col].clip(lower=lo, upper=hi)
        elif self.treatment == "nan":
            df.loc[mask, col] = np.nan
        elif self.treatment == "drop":
            df = df[~mask].copy()
        else:
            raise ValueError(f"Unknown treatment '{self.treatment}'.")

        logger.debug(
            f"[{station_id}] '{col}': {count} outliers ({pct:.1f}%) → "
            f"{self.treatment}d to [{lo:.2f}, {hi:.2f}]."
        )
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_numeric_cols(self, df: pd.DataFrame) -> List[str]:
        """Return numeric columns present in ``df`` that are in the treatment list."""
        return [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c in _TREATMENT_COLS or c in [
                s.name for s in CANONICAL_SCHEMA if s.dtype == "float64"
            ]
        ]

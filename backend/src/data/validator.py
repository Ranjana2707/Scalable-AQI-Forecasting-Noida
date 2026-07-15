"""
src/data/validator.py
======================
Schema and data-quality validation for AQI DataFrames.

Two complementary layers
------------------------
1. **Schema validation** (``SchemaValidator``): checks that required columns
   exist and have appropriate dtypes.
2. **Quality validation** (``QualityValidator``): checks completeness,
   value ranges, and statistical sanity of the data.

Usage
-----
    from src.data.validator import SchemaValidator, QualityValidator

    schema_v = SchemaValidator(cfg)
    schema_v.validate(df)           # raises on failure, returns df on success

    quality_v = QualityValidator(cfg)
    report = quality_v.validate(df) # returns a QualityReport dataclass
    report.print_summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# AQI value bounds (CPCB scale: 0–500)
# ---------------------------------------------------------------------------
AQI_MIN: float = 0.0
AQI_MAX: float = 500.0

POLLUTANT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "PM2.5":   (0.0,  1000.0),
    "PM10":    (0.0,  1500.0),
    "NO":      (0.0,   500.0),
    "NO2":     (0.0,   500.0),
    "NOx":     (0.0,  1000.0),
    "NH3":     (0.0,   500.0),
    "CO":      (0.0,    50.0),
    "SO2":     (0.0,   500.0),
    "O3":      (0.0,   500.0),
    "Benzene": (0.0,   100.0),
    "Toluene": (0.0,   200.0),
    "Xylene":  (0.0,   200.0),
}

METEO_BOUNDS: Dict[str, Tuple[float, float]] = {
    "temperature":   (-5.0, 55.0),    # °C, realistic for Noida
    "humidity":      (0.0,  100.0),   # %
    "wind_speed":    (0.0,   50.0),   # m/s
    "wind_direction":(0.0,  360.0),   # degrees
    "rainfall":      (0.0,  500.0),   # mm/day
}


# ---------------------------------------------------------------------------
# Quality Report
# ---------------------------------------------------------------------------

@dataclass
class ColumnQuality:
    name: str
    missing_count: int
    missing_pct: float
    out_of_range_count: int
    out_of_range_pct: float
    min_val: Optional[float]
    max_val: Optional[float]
    mean_val: Optional[float]
    passed: bool


@dataclass
class QualityReport:
    """
    Holds per-column quality statistics and an overall pass/fail flag.
    """
    total_rows: int
    total_cols: int
    columns: List[ColumnQuality] = field(default_factory=list)
    overall_passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def print_summary(self) -> None:
        """Print a human-readable quality report to stdout."""
        print("\n" + "=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        print(f"Total rows : {self.total_rows:,}")
        print(f"Total cols : {self.total_cols}")
        print(f"Overall    : {'✅ PASSED' if self.overall_passed else '❌ FAILED'}")

        if self.errors:
            print("\n🔴 Errors:")
            for e in self.errors:
                print(f"  • {e}")

        if self.warnings:
            print("\n🟡 Warnings:")
            for w in self.warnings:
                print(f"  • {w}")

        print("\nColumn-level summary:")
        header = f"{'Column':<20} {'Missing%':>9} {'OutOfRange%':>12} {'Status':>8}"
        print(header)
        print("-" * 55)
        for cq in self.columns:
            status = "✅" if cq.passed else "❌"
            print(
                f"{cq.name:<20} {cq.missing_pct:>8.1f}% "
                f"{cq.out_of_range_pct:>11.1f}%  {status}"
            )
        print("=" * 60 + "\n")

    def to_dict(self) -> dict:
        """Serialise the report to a JSON-compatible dictionary."""
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Schema Validator
# ---------------------------------------------------------------------------

class SchemaValidator:
    """
    Checks that a DataFrame has the required columns and correct dtypes.

    Parameters
    ----------
    config : AppConfig
        Project configuration specifying expected column names.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._required_cols = (
            [config.data.datetime_col]
            + [config.data.target_col]
            + config.data.pollutant_cols
            + config.data.meteorological_cols
        )

    def validate(self, df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
        """
        Validate schema; raise ``ValueError`` on critical failures.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.
        strict : bool
            If True, raise on any missing column (including optional ones).
            If False, only raise on target + datetime column absence.

        Returns
        -------
        pd.DataFrame
            The input DataFrame (unchanged) if validation passes.

        Raises
        ------
        ValueError
            On missing critical columns or wrong datetime dtype.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Critical: datetime and target must be present
        for critical_col in [self.config.data.datetime_col, self.config.data.target_col]:
            if critical_col not in df.columns:
                errors.append(f"Critical column missing: '{critical_col}'")

        # Warn on missing optional columns
        for col in self._required_cols:
            if col not in df.columns:
                msg = f"Expected column missing: '{col}'"
                if strict:
                    errors.append(msg)
                else:
                    warnings.append(msg)

        # Datetime dtype check
        dt_col = self.config.data.datetime_col
        if dt_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
                errors.append(
                    f"Column '{dt_col}' must be datetime64, "
                    f"got {df[dt_col].dtype}."
                )

        for w in warnings:
            logger.warning(f"[SchemaValidator] {w}")
        for e in errors:
            logger.error(f"[SchemaValidator] {e}")

        if errors:
            raise ValueError(
                f"Schema validation failed with {len(errors)} error(s):\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

        logger.info("Schema validation passed.")
        return df


# ---------------------------------------------------------------------------
# Quality Validator
# ---------------------------------------------------------------------------

class QualityValidator:
    """
    Checks completeness, value ranges, and statistical sanity of the data.

    Parameters
    ----------
    config : AppConfig
        Project configuration.
    min_valid_pct : float
        Minimum fraction of non-missing values required per column (0–1).
        Defaults to ``config.preprocessing.min_valid_data_pct``.
    """

    def __init__(
        self,
        config: AppConfig,
        min_valid_pct: Optional[float] = None,
    ) -> None:
        self.config = config
        self.min_valid_pct = min_valid_pct or config.preprocessing.min_valid_data_pct

    def validate(self, df: pd.DataFrame) -> QualityReport:
        """
        Run all quality checks and return a ``QualityReport``.

        Checks performed
        ----------------
        1. Missing-value rate per column.
        2. Out-of-range values for pollutants, meteorological vars, and AQI.
        3. Temporal continuity (gap detection in hourly data).
        4. Monotonicity of the datetime index.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that has already passed schema validation.

        Returns
        -------
        QualityReport
        """
        n = len(df)
        report = QualityReport(total_rows=n, total_cols=len(df.columns))

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Build bounds lookup for all known columns
        all_bounds = {**POLLUTANT_BOUNDS, **METEO_BOUNDS}
        target_col = self.config.data.target_col
        all_bounds[target_col] = (AQI_MIN, AQI_MAX)

        # ── Per-column checks ────────────────────────────────────────────────
        for col in numeric_cols:
            missing_count = int(df[col].isna().sum())
            missing_pct = missing_count / n * 100
            valid_pct = 1.0 - missing_count / n
            col_passed = True

            # Missing rate
            if valid_pct < self.min_valid_pct:
                msg = (
                    f"Column '{col}' has {missing_pct:.1f}% missing values "
                    f"(threshold: {(1-self.min_valid_pct)*100:.0f}%). "
                    "Consider dropping or heavy imputation."
                )
                report.warnings.append(msg)
                logger.warning(msg)
                col_passed = False

            # Out-of-range
            oor_count = 0
            oor_pct = 0.0
            lo, hi = all_bounds.get(col, (None, None))
            if lo is not None and hi is not None:
                series = df[col].dropna()
                oor_mask = (series < lo) | (series > hi)
                oor_count = int(oor_mask.sum())
                oor_pct = oor_count / len(series) * 100 if len(series) > 0 else 0.0
                if oor_count > 0:
                    msg = (
                        f"Column '{col}': {oor_count} values outside "
                        f"[{lo}, {hi}] ({oor_pct:.2f}%)."
                    )
                    report.warnings.append(msg)
                    logger.warning(msg)

            stats = df[col].dropna()
            report.columns.append(
                ColumnQuality(
                    name=col,
                    missing_count=missing_count,
                    missing_pct=round(missing_pct, 2),
                    out_of_range_count=oor_count,
                    out_of_range_pct=round(oor_pct, 2),
                    min_val=round(float(stats.min()), 3) if len(stats) else None,
                    max_val=round(float(stats.max()), 3) if len(stats) else None,
                    mean_val=round(float(stats.mean()), 3) if len(stats) else None,
                    passed=col_passed,
                )
            )
            if not col_passed:
                report.overall_passed = False

        # ── Temporal continuity ─────────────────────────────────────────────
        dt_col = self.config.data.datetime_col
        if dt_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[dt_col]):
            self._check_temporal_continuity(df[dt_col], report)

        logger.info(
            f"Quality validation complete | "
            f"passed={report.overall_passed} | "
            f"warnings={len(report.warnings)} | "
            f"errors={len(report.errors)}"
        )
        return report

    def _check_temporal_continuity(
        self, dt_series: pd.Series, report: QualityReport
    ) -> None:
        """Detect large temporal gaps and non-monotonic timestamps."""
        if not dt_series.is_monotonic_increasing:
            report.warnings.append(
                "Datetime column is not monotonically increasing. "
                "Sort the data before preprocessing."
            )

        diffs = dt_series.diff().dropna()
        expected_freq = pd.Timedelta("1h")
        large_gaps = diffs[diffs > expected_freq * 3]
        if not large_gaps.empty:
            report.warnings.append(
                f"Detected {len(large_gaps)} temporal gap(s) > 3 hours. "
                f"Largest gap: {diffs.max()}. "
                "Interpolation will fill these automatically."
            )
            logger.warning(
                f"Temporal gaps detected: {len(large_gaps)} gap(s); "
                f"max gap = {diffs.max()}"
            )

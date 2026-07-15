"""
src/preprocessing/validator.py
================================
Schema validation and data-type coercion for AQI DataFrames.

Responsibilities
----------------
1. Define the **canonical AQI schema** — the single source of truth for
   column names, dtypes, and value bounds across the entire project.
2. Detect which canonical columns are present vs missing vs misnamed.
3. Coerce numeric columns to ``float64`` and the datetime column to
   ``datetime64[ns]``.
4. Validate that numeric values fall within physically plausible bounds.
5. Return a typed, schema-compliant DataFrame — or raise with a clear
   actionable error message.

Why this matters for AQI forecasting
--------------------------------------
Downstream feature engineering and model training assume specific column
names and dtypes.  A single misnamed column (e.g. ``"pm2.5"`` vs ``"pm25"``)
would silently produce NaN features.  This validator makes that contract
explicit and enforced at ingestion time.

Multi-station scalability
--------------------------
The canonical schema is defined once here.  Adding a new station requires
only that its data is mapped to this schema — no changes to any downstream
module.

Usage
-----
    from src.preprocessing.validator import SchemaValidator

    validator = SchemaValidator(config)
    df_valid  = validator.validate(df_raw)          # raises on critical errors
    report    = validator.get_validation_report()   # dict of findings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Canonical AQI Schema
# ---------------------------------------------------------------------------
# This is the single source of truth. Every column that enters the ML
# pipeline must conform to this schema.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnSpec:
    """Specification for one column in the canonical schema."""
    name: str                          # canonical name used throughout project
    dtype: str                         # target dtype: "datetime64", "float64", "str"
    required: bool                     # True = pipeline fails without it
    unit: str                          # for documentation / reports
    physical_min: Optional[float]      # None = no lower bound check
    physical_max: Optional[float]      # None = no upper bound check
    description: str


CANONICAL_SCHEMA: List[ColumnSpec] = [
    ColumnSpec("date",        "datetime64", True,  "datetime",  None,    None,   "Observation timestamp"),
    ColumnSpec("station",     "str",        True,  "id",        None,    None,   "Monitoring station identifier"),
    ColumnSpec("aqi",         "float64",    True,  "index",     0.0,     500.0,  "Air Quality Index (CPCB scale)"),
    ColumnSpec("pm25",        "float64",    False, "µg/m³",     0.0,     1000.0, "PM2.5 concentration"),
    ColumnSpec("pm10",        "float64",    False, "µg/m³",     0.0,     1500.0, "PM10 concentration"),
    ColumnSpec("no2",         "float64",    False, "µg/m³",     0.0,     500.0,  "Nitrogen Dioxide"),
    ColumnSpec("so2",         "float64",    False, "µg/m³",     0.0,     500.0,  "Sulphur Dioxide"),
    ColumnSpec("co",          "float64",    False, "mg/m³",     0.0,     50.0,   "Carbon Monoxide"),
    ColumnSpec("o3",          "float64",    False, "µg/m³",     0.0,     500.0,  "Ozone"),
    ColumnSpec("nh3",         "float64",    False, "µg/m³",     0.0,     500.0,  "Ammonia"),
    ColumnSpec("temperature", "float64",    False, "°C",        -5.0,    55.0,   "Ambient temperature"),
    ColumnSpec("humidity",    "float64",    False, "%",          0.0,    100.0,  "Relative humidity"),
    ColumnSpec("wind_speed",  "float64",    False, "m/s",        0.0,    50.0,   "Wind speed"),
    ColumnSpec("pressure",    "float64",    False, "hPa",       800.0,   1100.0, "Atmospheric pressure"),
]

# Quick lookup: canonical_name → ColumnSpec
SCHEMA_MAP: Dict[str, ColumnSpec] = {spec.name: spec for spec in CANONICAL_SCHEMA}

# Common raw column aliases → canonical name.
# Add new aliases here without touching any other file.
COLUMN_ALIASES: Dict[str, str] = {
    # Datetime variants
    "datetime":             "date",
    "timestamp":            "date",
    "date_time":            "date",
    "time":                 "date",
    "Date":                 "date",
    "Timestamp":            "date",

    # AQI
    "AQI":                  "aqi",
    "aqi_value":            "aqi",
    "air_quality_index":    "aqi",

    # PM2.5
    "PM2.5":                "pm25",
    "pm2.5":                "pm25",
    "PM25":                 "pm25",
    "pm_2_5":               "pm25",
    "PM2_5":                "pm25",

    # PM10
    "PM10":                 "pm10",
    "PM_10":                "pm10",

    # Gases
    "NO2":                  "no2",
    "SO2":                  "so2",
    "CO":                   "co",
    "O3":                   "o3",
    "NH3":                  "nh3",

    # Meteorology
    "temp":                 "temperature",
    "Temp":                 "temperature",
    "Temperature":          "temperature",
    "RH":                   "humidity",
    "relative_humidity":    "humidity",
    "Humidity":             "humidity",
    "WS":                   "wind_speed",
    "wind speed":           "wind_speed",
    "WindSpeed":            "wind_speed",
    "BP":                   "pressure",
    "atm_pressure":         "pressure",
    "Pressure":             "pressure",
    "Station":              "station",
    "station_id":           "station",
    "StationName":          "station",
}


# ---------------------------------------------------------------------------
# Validation result container
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Structured record of all schema validation findings."""
    station_id: str
    total_rows: int
    total_cols_raw: int

    present_columns: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    renamed_columns: Dict[str, str] = field(default_factory=dict)   # old → new
    dtype_coercions: Dict[str, str] = field(default_factory=dict)   # col → new dtype
    range_violations: Dict[str, int] = field(default_factory=dict)  # col → count
    unknown_columns: List[str] = field(default_factory=list)
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def print_summary(self) -> None:
        """Print a human-readable validation summary."""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        print(f"\n{'='*60}")
        print(f"SCHEMA VALIDATION — {self.station_id}")
        print(f"{'='*60}")
        print(f"Status : {status}")
        print(f"Rows   : {self.total_rows:,}")
        print(f"Present columns  : {self.present_columns}")
        if self.missing_required:
            print(f"🔴 Missing required : {self.missing_required}")
        if self.missing_optional:
            print(f"🟡 Missing optional : {self.missing_optional}")
        if self.renamed_columns:
            print(f"🔄 Renamed          : {self.renamed_columns}")
        if self.dtype_coercions:
            print(f"🔄 Type coercions   : {self.dtype_coercions}")
        if self.range_violations:
            print(f"⚠️  Range violations : {self.range_violations}")
        if self.unknown_columns:
            print(f"ℹ️  Unknown columns  : {self.unknown_columns}")
        if self.errors:
            print("Errors:")
            for e in self.errors:
                print(f"  • {e}")
        print(f"{'='*60}\n")

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# SchemaValidator
# ---------------------------------------------------------------------------

class SchemaValidator:
    """
    Validates and normalises a raw DataFrame against the canonical AQI schema.

    Parameters
    ----------
    config : AppConfig
        Project configuration.

    Attributes
    ----------
    _report : ValidationReport | None
        Populated after ``validate()`` is called.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._report: Optional[ValidationReport] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        df: pd.DataFrame,
        station_id: Optional[str] = None,
        strict: bool = False,
    ) -> pd.DataFrame:
        """
        Validate ``df`` against the canonical schema and return a normalised copy.

        Steps performed
        ---------------
        1. Rename aliased columns to their canonical names.
        2. Check for missing required columns → error.
        3. Check for missing optional columns → warning (column added as NaN).
        4. Coerce numeric columns to ``float64``.
        5. Coerce datetime column to ``datetime64[ns]``.
        6. Detect out-of-physical-range values → warning (values not modified here).
        7. Log unknown columns (kept as-is for traceability).

        Parameters
        ----------
        df : pd.DataFrame
            Raw or partially normalised DataFrame.
        station_id : str, optional
            Overrides ``config.project.station_id`` for logging.
        strict : bool
            If True, treat missing optional columns as errors.

        Returns
        -------
        pd.DataFrame
            Schema-compliant DataFrame.  All canonical columns that were
            present in the raw data are renamed, typed, and validated.
            Optional columns not in the raw data are added as ``NaN``.

        Raises
        ------
        ValueError
            If any required column is missing or cannot be coerced.
        """
        sid = station_id or self.config.project.station_id
        df = df.copy()

        self._report = ValidationReport(
            station_id=sid,
            total_rows=len(df),
            total_cols_raw=len(df.columns),
        )

        # Step 1 — rename aliases
        df = self._rename_aliases(df)

        # Step 2 — check required columns
        self._check_required_columns(df, strict)

        # Step 3 — add missing optional columns as NaN
        df = self._add_missing_optional_columns(df)

        # Step 4 — coerce dtypes
        df = self._coerce_dtypes(df, sid)

        # Step 5 — range checks (log only; cleaning happens in cleaner.py)
        self._check_value_ranges(df)

        # Step 6 — identify unknown columns
        canonical_names = {spec.name for spec in CANONICAL_SCHEMA}
        self._report.unknown_columns = [
            c for c in df.columns if c not in canonical_names
        ]
        if self._report.unknown_columns:
            logger.info(
                f"[{sid}] Unknown columns kept as-is: "
                f"{self._report.unknown_columns}"
            )

        # Final pass — determine present canonical columns
        self._report.present_columns = [
            c for c in df.columns if c in canonical_names
        ]

        if self._report.errors:
            self._report.passed = False
            raise ValueError(
                f"Schema validation failed for station '{sid}':\n"
                + "\n".join(f"  • {e}" for e in self._report.errors)
            )

        logger.info(
            f"[{sid}] Schema validation passed | "
            f"present={self._report.present_columns} | "
            f"missing_optional={self._report.missing_optional}"
        )
        return df

    def get_validation_report(self) -> ValidationReport:
        """
        Return the most recent validation report.

        Returns
        -------
        ValidationReport

        Raises
        ------
        RuntimeError
            If called before ``validate()``.
        """
        if self._report is None:
            raise RuntimeError("Call validate() before get_validation_report().")
        return self._report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rename_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply COLUMN_ALIASES mapping to standardise column names."""
        rename_map = {}
        for col in df.columns:
            canonical = COLUMN_ALIASES.get(col)
            if canonical and col != canonical:
                rename_map[col] = canonical
        if rename_map:
            df = df.rename(columns=rename_map)
            self._report.renamed_columns = rename_map
            logger.debug(f"Renamed columns: {rename_map}")
        return df

    def _check_required_columns(
        self, df: pd.DataFrame, strict: bool
    ) -> None:
        """Flag missing required columns as errors."""
        for spec in CANONICAL_SCHEMA:
            if spec.required and spec.name not in df.columns:
                msg = (
                    f"Required column '{spec.name}' ({spec.description}) "
                    f"is missing. Add it to the raw data or COLUMN_ALIASES."
                )
                self._report.errors.append(msg)
                logger.error(msg)
            elif not spec.required and spec.name not in df.columns:
                msg = (
                    f"Optional column '{spec.name}' ({spec.description}) "
                    f"not present; will be filled with NaN."
                )
                self._report.missing_optional.append(spec.name)
                if strict:
                    self._report.errors.append(msg)
                else:
                    self._report.warnings.append(msg)
                    logger.warning(msg)

    def _add_missing_optional_columns(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add every optional canonical column that is absent as NaN float64."""
        for spec in CANONICAL_SCHEMA:
            if not spec.required and spec.name not in df.columns:
                if spec.dtype == "float64":
                    df[spec.name] = np.nan
                    logger.debug(f"Added missing optional column '{spec.name}' as NaN.")
        return df

    def _coerce_dtypes(
        self, df: pd.DataFrame, station_id: str
    ) -> pd.DataFrame:
        """
        Coerce each canonical column to its target dtype.

        Non-numeric strings in numeric columns are converted to NaN
        (this is intentional — coercion errors are handled gracefully).
        """
        for spec in CANONICAL_SCHEMA:
            if spec.name not in df.columns:
                continue

            col = spec.name
            current_dtype = str(df[col].dtype)

            if spec.dtype == "float64":
                if not pd.api.types.is_float_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    self._report.dtype_coercions[col] = "float64"
                    coerced_nan = df[col].isna().sum()
                    if coerced_nan > 0:
                        logger.warning(
                            f"[{station_id}] '{col}': {coerced_nan} values "
                            f"could not be coerced to float64 → set to NaN."
                        )

            elif spec.dtype == "datetime64":
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    self._report.dtype_coercions[col] = "datetime64[ns]"
                    nat_count = df[col].isna().sum()
                    if nat_count > 0:
                        logger.warning(
                            f"[{station_id}] '{col}': {nat_count} values "
                            f"could not be parsed as datetime → NaT."
                        )

            elif spec.dtype == "str":
                df[col] = df[col].astype(str)

        return df

    def _check_value_ranges(self, df: pd.DataFrame) -> None:
        """
        Log out-of-physical-range values. Does NOT modify data
        (that is the OutlierHandler's responsibility).
        """
        for spec in CANONICAL_SCHEMA:
            if spec.dtype != "float64":
                continue
            if spec.name not in df.columns:
                continue
            if spec.physical_min is None and spec.physical_max is None:
                continue

            series = df[spec.name].dropna()
            lo = spec.physical_min if spec.physical_min is not None else -np.inf
            hi = spec.physical_max if spec.physical_max is not None else np.inf
            violations = int(((series < lo) | (series > hi)).sum())

            if violations > 0:
                pct = violations / len(series) * 100
                msg = (
                    f"Column '{spec.name}': {violations} values ({pct:.1f}%) "
                    f"outside physical bounds [{spec.physical_min}, "
                    f"{spec.physical_max}] {spec.unit}."
                )
                self._report.range_violations[spec.name] = violations
                self._report.warnings.append(msg)
                logger.warning(msg)

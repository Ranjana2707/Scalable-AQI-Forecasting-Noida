"""
src/pdf_ingestion/schema_mapper.py
=====================================
Maps raw extracted rows (list-of-lists) to the project's canonical
DataFrame schema defined in ``src/preprocessing/validator.py``.

Responsibilities
----------------
1. Assign canonical column names based on detected PDF type.
2. Parse dates to ``datetime64``.
3. Coerce all numeric columns to ``float64``.
4. Add the ``station`` column.
5. Return a clean ``pd.DataFrame`` with canonical column names.

Column mapping tables
---------------------
UPPCB_ANNUAL (11 cols after month strip):
    0: date  1:aqi  2:pm25  3:pm10  4:no2  5:so2  6:co  7:o3  8:nh3  9:pb   10:(extra)

CAAQMS_METEO (15 cols):
    0:date  1:aqi  2:category(drop)  3:pm25  4:pm10  5:no2  6:nh3  7:so2  8:co  9:o3
    10:pb   11:temperature  12:humidity  13:wind_speed  14:wind_direction  15:pressure

UPPCB_SHORT (10 cols):
    0:date  1:aqi  2:pm25  3:pm10  4:no2  5:so2  6:co  7:o3  8:nh3  9:pb

Usage
-----
    mapper = SchemaMapper()
    df = mapper.map(rows, pdf_type="UPPCB_ANNUAL", station_id="noida_sector_1")
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column mapping tables — raw position → canonical name
# ---------------------------------------------------------------------------

_UPPCB_ANNUAL_COLS: Dict[int, str] = {
    0: "date",
    1: "aqi",
    2: "pm25",
    3: "pm10",
    4: "no2",
    5: "so2",
    6: "co",
    7: "o3",
    8: "nh3",
    9: "pb",
}

_CAAQMS_METEO_COLS: Dict[int, str] = {
    0:  "date",
    1:  "aqi",
    2:  "_category",       # AQI category text — kept for validation, dropped later
    3:  "pm25",
    4:  "pm10",
    5:  "no2",
    6:  "nh3",
    7:  "so2",
    8:  "co",
    9:  "o3",
    10: "pb",
    11: "temperature",
    12: "humidity",
    13: "wind_speed",
    14: "wind_direction",
}
# pressure appears at col 15 in the CAAQMS data (15-col type)
# But the extractor normalises to 15 cols, so we handle it here:
_CAAQMS_METEO_COLS[15] = "pressure"   # index 15 if 16 total cols extracted

_UPPCB_SHORT_COLS: Dict[int, str] = {
    0: "date",
    1: "aqi",
    2: "pm25",
    3: "pm10",
    4: "no2",
    5: "so2",
    6: "co",
    7: "o3",
    8: "nh3",
    9: "pb",
}

_COL_MAPS = {
    "UPPCB_ANNUAL": _UPPCB_ANNUAL_COLS,
    "CAAQMS_METEO": _CAAQMS_METEO_COLS,
    "UPPCB_SHORT":  _UPPCB_SHORT_COLS,
}

# Columns to drop after mapping (internal / redundant)
_DROP_COLS = {"_category"}

# Canonical numeric columns
_NUMERIC_COLS = {
    "aqi", "pm25", "pm10", "no2", "so2", "co", "o3", "nh3", "pb",
    "temperature", "humidity", "wind_speed", "wind_direction", "pressure",
}


class SchemaMapper:
    """
    Maps raw extracted rows to the canonical AQI DataFrame schema.

    Parameters
    ----------
    None — stateless, safe to share across pipeline instances.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map(
        self,
        rows: List[List[Optional[str]]],
        pdf_type: str,
        station_id: str,
    ) -> pd.DataFrame:
        """
        Convert a list-of-lists into a canonically-named, typed DataFrame.

        Parameters
        ----------
        rows : list of list
            Raw rows from ``TableExtractor.merge_and_clean()``.
        pdf_type : str
            One of ``UPPCB_ANNUAL``, ``CAAQMS_METEO``, ``UPPCB_SHORT``.
        station_id : str
            Station identifier attached as a ``station`` column.

        Returns
        -------
        pd.DataFrame
            Columns: date (datetime64), station (str), aqi, pm25, pm10, …
            Numeric columns are float64; missing values are NaN.

        Raises
        ------
        ValueError
            If ``pdf_type`` is not recognised.
        """
        if pdf_type not in _COL_MAPS:
            raise ValueError(
                f"Unknown pdf_type '{pdf_type}'. "
                f"Choose from: {list(_COL_MAPS.keys())}"
            )

        col_map = _COL_MAPS[pdf_type]
        logger.info(
            f"[{station_id}] Mapping {len(rows)} rows | type={pdf_type}"
        )

        # Step 1: build DataFrame with positional column names
        df = pd.DataFrame(rows)

        # Step 2: rename columns using the map
        rename = {pos: name for pos, name in col_map.items() if pos < len(df.columns)}
        df = df.rename(columns=rename)

        # Step 3: keep only mapped columns + drop internal ones
        keep_cols = [
            col_map[i] for i in sorted(col_map.keys())
            if i < len(df.columns) and col_map[i] not in _DROP_COLS
        ]
        # Deduplicate while preserving order
        seen: set = set()
        keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]
        df = df[[c for c in keep_cols if c in df.columns]]

        # Step 4: attach station column (always at position 1 after date)
        df.insert(1, "station", station_id)

        # Step 5: parse date
        df = self._parse_date(df, station_id)

        # Step 6: coerce numerics
        df = self._coerce_numerics(df, station_id)

        # Step 7: validate range — log only
        self._log_range_warnings(df, station_id)

        logger.info(
            f"[{station_id}] Schema mapped | shape={df.shape} | "
            f"cols={df.columns.tolist()}"
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_date(self, df: pd.DataFrame, station_id: str) -> pd.DataFrame:
        """Parse the ``date`` column to datetime64, coerce errors to NaT."""
        if "date" not in df.columns:
            logger.error(f"[{station_id}] 'date' column not found after mapping.")
            return df
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        nat_count = df["date"].isna().sum()
        if nat_count:
            logger.warning(
                f"[{station_id}] {nat_count} date(s) could not be parsed → NaT."
            )
        # Drop rows where date is NaT (completely unrecoverable)
        df = df.dropna(subset=["date"]).reset_index(drop=True)
        return df

    def _coerce_numerics(
        self, df: pd.DataFrame, station_id: str
    ) -> pd.DataFrame:
        """
        Coerce numeric canonical columns to float64.
        Non-numeric strings become NaN (e.g., '--', 'N/A', blank).
        """
        for col in df.columns:
            if col in _NUMERIC_COLS:
                before_nulls = df[col].isna().sum()
                df[col] = pd.to_numeric(df[col], errors="coerce")
                after_nulls = df[col].isna().sum()
                new_nulls = after_nulls - before_nulls
                if new_nulls > 0:
                    logger.debug(
                        f"[{station_id}] '{col}': {new_nulls} value(s) "
                        "could not be coerced → NaN."
                    )
        return df

    @staticmethod
    def _log_range_warnings(df: pd.DataFrame, station_id: str) -> None:
        """Log AQI values outside the CPCB 0–500 range (informational only)."""
        if "aqi" in df.columns:
            out = df[(df["aqi"] < 0) | (df["aqi"] > 500)]["aqi"]
            if not out.empty:
                logger.warning(
                    f"[{station_id}] {len(out)} AQI value(s) outside [0, 500]: "
                    f"min={out.min():.0f}, max={out.max():.0f}"
                )

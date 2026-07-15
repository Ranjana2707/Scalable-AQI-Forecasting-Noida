"""
src/preprocessing/loader.py
============================
Multi-format data ingestion for AQI monitoring station data.

Supported formats
-----------------
- CSV  (.csv)                — primary format from CPCB portal
- Excel (.xlsx, .xls)       — common for government data exports
- PDF  (.pdf)                — future support via extraction hook

Design for multi-station scalability
--------------------------------------
``DataLoader`` is stateless with respect to station identity.  Every public
method accepts an optional ``station_id`` parameter that overrides the value
from config.  A multi-station orchestrator therefore simply calls::

    for sid in station_ids:
        df = loader.load(station_id=sid)

without any other changes.

Usage
-----
    from src.preprocessing.loader import DataLoader
    from src.utils.config import load_config

    cfg  = load_config("configs/default.yaml")
    loader = DataLoader(cfg)
    df   = loader.load()                          # single station from config
    df   = loader.load(station_id="noida_s125")  # override for multi-station
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Candidate datetime formats tried in order (CPCB portal variations)
_DATETIME_FORMATS: List[str] = [
    "%Y-%m-%d %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%d/%m/%Y %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d",
    "%d-%m-%Y",
]


class DataLoader:
    """
    Discovers and loads raw AQI data files for a monitoring station.

    Responsibilities
    ----------------
    1. Discover all files matching ``<station_id>*.{csv,xlsx,xls}`` in the
       raw data directory.
    2. Dispatch each file to the correct format reader.
    3. Normalise column names (strip whitespace, consistent casing).
    4. Parse and standardise the datetime column.
    5. Concatenate multi-file datasets into one DataFrame.
    6. Attach a ``station`` column so provenance is always traceable.

    Parameters
    ----------
    config : AppConfig
        Loaded project configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.raw_dir: Path = Path(config.paths.raw_data)
        self.dt_col: str = config.data.datetime_col
        logger.info(
            f"DataLoader initialised | raw_dir={self.raw_dir}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        station_id: Optional[str] = None,
        file_pattern: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Discover and load all data files for a station, returning one
        chronologically-sorted, deduplicated DataFrame.

        Parameters
        ----------
        station_id : str, optional
            Station identifier.  Defaults to ``config.project.station_id``.
            Pass explicitly to support multi-station loops.
        file_pattern : str, optional
            Custom glob pattern relative to ``raw_dir``.  If omitted, the
            pattern ``<station_id>*.(csv|xlsx|xls)`` is used.

        Returns
        -------
        pd.DataFrame
            Raw data with:
            - A parsed ``datetime`` column (not yet the index).
            - A ``station`` column containing ``station_id``.
            - Normalised column names (stripped whitespace).

        Raises
        ------
        FileNotFoundError
            If no files are found and the raw data directory is empty.
        ValueError
            If the datetime column cannot be parsed in any known format.
        """
        sid = station_id or self.config.project.station_id
        files = self._discover_files(sid, file_pattern)
        logger.info(f"[{sid}] Found {len(files)} file(s): {[Path(f).name for f in files]}")

        frames: List[pd.DataFrame] = []
        for fp in files:
            df_chunk = self._read_file(fp, sid)
            frames.append(df_chunk)

        df = pd.concat(frames, axis=0, ignore_index=True)
        df = self._normalise_columns(df)
        df = self._parse_datetime_column(df, sid)
        df = self._attach_station_column(df, sid)
        df = self._sort_and_deduplicate(df, sid)

        logger.info(
            f"[{sid}] Load complete | rows={len(df):,} | cols={df.columns.tolist()} | "
            f"range={df[self.dt_col].min()} → {df[self.dt_col].max()}"
        )
        return df

    def load_external(self, filename: str) -> pd.DataFrame:
        """
        Load a file from ``data/external/`` (holidays, weather reanalysis, etc.).

        Parameters
        ----------
        filename : str
            Filename inside ``data/external/``.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        src = Path(self.config.paths.external_data) / filename
        if not src.exists():
            raise FileNotFoundError(f"External file not found: {src}")
        df = self._read_file(str(src), station_id="external")
        logger.info(f"External data loaded: {src} | shape={df.shape}")
        return df

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _discover_files(
        self, station_id: str, file_pattern: Optional[str]
    ) -> List[str]:
        """
        Glob the raw directory for all files belonging to ``station_id``.

        Tries CSV, then Excel.  If a custom ``file_pattern`` is supplied it
        takes precedence.

        Returns
        -------
        list of str
            Sorted list of absolute file paths.

        Raises
        ------
        FileNotFoundError
            If no files are found.
        """
        if file_pattern:
            found = sorted(glob.glob(str(self.raw_dir / file_pattern)))
        else:
            # Try station-prefixed files first, then fall back to any file in dir
            patterns = [
                f"{station_id}*.csv",
                f"{station_id}*.xlsx",
                f"{station_id}*.xls",
            ]
            found = []
            for pat in patterns:
                found.extend(glob.glob(str(self.raw_dir / pat)))
            found = sorted(set(found))

        if not found:
            # Development fallback: pick up any CSV/Excel in the raw dir
            fallback: List[str] = []
            for ext in ("*.csv", "*.xlsx", "*.xls"):
                fallback.extend(glob.glob(str(self.raw_dir / ext)))
            fallback = sorted(set(fallback))

            if fallback:
                logger.warning(
                    f"[{station_id}] No files matched pattern for station. "
                    f"Falling back to ALL files in {self.raw_dir}: "
                    f"{[Path(f).name for f in fallback]}"
                )
                return fallback

            raise FileNotFoundError(
                f"No data files found for station '{station_id}' in {self.raw_dir}.\n"
                f"Expected files matching: <station_id>*.csv / *.xlsx\n"
                f"Place your raw data in: {self.raw_dir.resolve()}"
            )

        return found

    # ------------------------------------------------------------------
    # Format readers
    # ------------------------------------------------------------------

    def _read_file(self, filepath: str, station_id: str) -> pd.DataFrame:
        """
        Dispatch a single file to the appropriate format reader.

        Parameters
        ----------
        filepath : str
            Absolute path to the data file.
        station_id : str
            Used only for log context.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        ext = Path(filepath).suffix.lower()
        logger.debug(f"[{station_id}] Reading {ext} file: {Path(filepath).name}")

        if ext == ".csv":
            return self._read_csv(filepath)
        elif ext in (".xlsx", ".xls"):
            return self._read_excel(filepath)
        elif ext == ".pdf":
            return self._read_pdf(filepath)
        else:
            raise ValueError(
                f"Unsupported file format '{ext}' for file: {filepath}.\n"
                f"Supported: .csv, .xlsx, .xls  |  Future: .pdf"
            )

    def _read_csv(self, filepath: str) -> pd.DataFrame:
        """
        Read a CSV file with robust encoding and separator detection.

        Attempts UTF-8 first; falls back to latin-1 for legacy CPCB exports.
        Auto-detects comma vs semicolon separator.
        """
        try:
            df = pd.read_csv(filepath, encoding="utf-8", sep=None,
                             engine="python", low_memory=False)
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decode failed for {filepath}; retrying with latin-1.")
            df = pd.read_csv(filepath, encoding="latin-1", sep=None,
                             engine="python", low_memory=False)
        logger.debug(f"CSV loaded: {Path(filepath).name} | shape={df.shape}")
        return df

    def _read_excel(self, filepath: str) -> pd.DataFrame:
        """
        Read the first sheet of an Excel file.

        Uses ``openpyxl`` for .xlsx and ``xlrd`` for legacy .xls.
        """
        ext = Path(filepath).suffix.lower()
        engine = "openpyxl" if ext == ".xlsx" else "xlrd"
        try:
            df = pd.read_excel(filepath, engine=engine, sheet_name=0)
        except Exception as exc:
            # Fallback: try without specifying engine
            logger.warning(
                f"Excel read with engine='{engine}' failed ({exc}); "
                "retrying without engine hint."
            )
            df = pd.read_excel(filepath, sheet_name=0)
        logger.debug(f"Excel loaded: {Path(filepath).name} | shape={df.shape}")
        return df

    def _read_pdf(self, filepath: str) -> pd.DataFrame:
        """
        PDF ingestion hook — not yet implemented.

        Future implementation will use ``pdfplumber`` or ``camelot`` to
        extract tabular data from CPCB bulletin PDFs.

        Raises
        ------
        NotImplementedError
            Always, until Phase 2 extension.
        """
        raise NotImplementedError(
            "PDF ingestion is planned but not yet implemented.\n"
            "Upcoming: pdfplumber-based table extraction for CPCB bulletins.\n"
            f"File: {filepath}"
        )

    # ------------------------------------------------------------------
    # Post-load normalisations
    # ------------------------------------------------------------------

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strip leading/trailing whitespace from all column names.

        Column values in string columns are also stripped to avoid
        invisible characters that break downstream type conversion.
        """
        df.columns = [str(c).strip() for c in df.columns]
        # Strip string cell values
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()
        return df

    def _parse_datetime_column(
        self, df: pd.DataFrame, station_id: str
    ) -> pd.DataFrame:
        """
        Parse the datetime column to ``datetime64[ns]``.

        Tries ``pd.to_datetime`` with ``infer_datetime_format=True`` first
        (fastest), then falls back through ``_DATETIME_FORMATS``.

        Raises
        ------
        ValueError
            If all format attempts fail.
        """
        col = self.dt_col
        if col not in df.columns:
            # Surface a helpful error listing what columns exist
            raise ValueError(
                f"[{station_id}] Datetime column '{col}' not found.\n"
                f"Available columns: {df.columns.tolist()}\n"
                f"Set 'data.datetime_col' in configs/default.yaml to the "
                f"correct column name."
            )

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            logger.debug(f"[{station_id}] '{col}' already datetime64 — no parsing needed.")
            return df

        # Fast path
        try:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
            logger.debug(f"[{station_id}] Datetime parsed via infer_datetime_format.")
            return df
        except Exception:
            pass

        # Explicit format fallback
        for fmt in _DATETIME_FORMATS:
            try:
                df[col] = pd.to_datetime(df[col], format=fmt)
                logger.debug(f"[{station_id}] Datetime parsed with format '{fmt}'.")
                return df
            except Exception:
                continue

        raise ValueError(
            f"[{station_id}] Cannot parse datetime column '{col}'.\n"
            f"Sample values: {df[col].head(5).tolist()}\n"
            f"Tried formats: {_DATETIME_FORMATS}\n"
            f"Add the correct format to _DATETIME_FORMATS in loader.py."
        )

    def _attach_station_column(
        self, df: pd.DataFrame, station_id: str
    ) -> pd.DataFrame:
        """
        Ensure a ``station`` column exists and is populated.

        If the raw file already contains a ``station`` column (common in
        multi-station consolidated exports) it is preserved.  Otherwise it
        is created from ``station_id``.

        This column is the key to multi-station scalability: every row
        always knows which station produced it.
        """
        if "station" not in df.columns:
            df.insert(1, "station", station_id)
            logger.debug(f"Station column added: '{station_id}'")
        else:
            # Normalise existing values
            df["station"] = df["station"].astype(str).str.strip()
        return df

    def _sort_and_deduplicate(
        self, df: pd.DataFrame, station_id: str
    ) -> pd.DataFrame:
        """
        Sort by datetime ascending and remove exact duplicate timestamps.

        For duplicates, the first occurrence is kept (conservative default).
        """
        before = len(df)
        df = df.sort_values(self.dt_col).reset_index(drop=True)
        df = df.drop_duplicates(subset=[self.dt_col, "station"], keep="first")
        df = df.reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.warning(
                f"[{station_id}] Removed {dropped:,} duplicate "
                f"(datetime, station) rows."
            )
        return df

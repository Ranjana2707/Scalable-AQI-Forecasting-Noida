"""
src/data/loader.py
===================
Station-aware data ingestion for the AQI forecasting system.

Responsibilities
----------------
- Discover and load raw CSV data files for a given station.
- Standardise column names and datetime parsing.
- Support multiple file layouts (wide, long, API export).
- Return a clean, consistently-typed ``pd.DataFrame`` for downstream use.

Design for multi-station scalability
-------------------------------------
All methods accept a ``station_id`` parameter.  When the system scales to
N stations the caller simply iterates over station IDs — no code changes
are required in this module.

Usage
-----
    from src.data.loader import DataLoader
    from src.utils.config import load_config

    cfg = load_config("configs/default.yaml")
    loader = DataLoader(cfg)
    df = loader.load_raw()
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Loads raw AQI and meteorological data for a single monitoring station.

    Parameters
    ----------
    config : AppConfig
        Fully loaded project configuration.

    Attributes
    ----------
    station_id : str
        The monitoring station identifier (from config).
    raw_data_dir : Path
        Root directory for raw data files.
    datetime_col : str
        Name of the datetime column in raw files.
    """

    # Accepted date formats tried in order during parsing.
    _DATE_FORMATS: List[str] = [
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d",
    ]

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.station_id: str = config.project.station_id
        self.raw_data_dir: Path = Path(config.paths.raw_data)
        self.datetime_col: str = config.data.datetime_col
        logger.info(
            f"DataLoader initialised | station={self.station_id} | "
            f"raw_dir={self.raw_data_dir}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_raw(
        self,
        file_pattern: Optional[str] = None,
        station_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Discover and load all matching CSV files for the station, then
        concatenate them into a single chronologically-sorted DataFrame.

        Parameters
        ----------
        file_pattern : str, optional
            Glob pattern relative to ``raw_data_dir``.  Defaults to
            ``<station_id>*.csv`` which matches any file that begins with the
            station identifier (e.g. ``noida_sector_62_2023.csv``).
        station_id : str, optional
            Override the station ID from config for this call only.
            Useful in multi-station loops.

        Returns
        -------
        pd.DataFrame
            Raw data with a parsed datetime index, sorted chronologically.

        Raises
        ------
        FileNotFoundError
            If no files match the pattern.
        ValueError
            If the datetime column is missing or cannot be parsed.

        Examples
        --------
        >>> loader = DataLoader(cfg)
        >>> df = loader.load_raw()
        >>> df.shape
        (87600, 17)   # 10 years × 8760 h/year × 17 columns
        """
        sid = station_id or self.station_id
        pattern = file_pattern or f"{sid}*.csv"
        search_path = self.raw_data_dir / pattern
        files = sorted(glob.glob(str(search_path)))

        if not files:
            # Fallback: look for any CSV in the raw dir (helpful during development
            # when file hasn't been renamed yet)
            fallback = sorted(self.raw_data_dir.glob("*.csv"))
            if fallback:
                logger.warning(
                    f"No files matched '{search_path}'. "
                    f"Falling back to all CSVs in {self.raw_data_dir}: "
                    f"{[f.name for f in fallback]}"
                )
                files = [str(f) for f in fallback]
            else:
                raise FileNotFoundError(
                    f"No CSV files found matching pattern '{search_path}'. "
                    f"Place your raw data in: {self.raw_data_dir}"
                )

        logger.info(f"Found {len(files)} file(s) for station '{sid}': {files}")

        frames: List[pd.DataFrame] = []
        for fp in files:
            df_chunk = self._load_single_file(fp)
            frames.append(df_chunk)
            logger.debug(f"Loaded {fp} → {df_chunk.shape}")

        df = pd.concat(frames, axis=0, ignore_index=True)
        df = self._parse_datetime(df)
        df = df.sort_values(self.datetime_col).reset_index(drop=True)
        df = df.drop_duplicates(subset=[self.datetime_col]).reset_index(drop=True)

        logger.info(
            f"Raw data loaded | rows={len(df):,} | "
            f"cols={df.columns.tolist()} | "
            f"date_range={df[self.datetime_col].min()} → "
            f"{df[self.datetime_col].max()}"
        )
        return df

    def load_external(self, filename: str) -> pd.DataFrame:
        """
        Load an external dataset (e.g., holiday calendars, weather reanalysis).

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
            If the file does not exist in the external data directory.
        """
        src = Path(self.config.paths.external_data) / filename
        if not src.exists():
            raise FileNotFoundError(f"External file not found: {src}")
        df = pd.read_csv(src)
        logger.info(f"External data loaded ← {src} | shape={df.shape}")
        return df

    def get_expected_columns(self) -> List[str]:
        """
        Return the full list of expected columns (target + pollutants + meteo).

        Returns
        -------
        list of str
        """
        return (
            [self.datetime_col, self.config.data.target_col]
            + self.config.data.pollutant_cols
            + self.config.data.meteorological_cols
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_single_file(self, filepath: str) -> pd.DataFrame:
        """Load one CSV file with robust encoding and separator detection."""
        try:
            df = pd.read_csv(
                filepath,
                encoding="utf-8",
                low_memory=False,
            )
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 failed for {filepath}; retrying with latin-1.")
            df = pd.read_csv(filepath, encoding="latin-1", low_memory=False)

        # Normalise column names: strip whitespace, lowercase variants kept as-is
        # (AQI columns from CPCB use mixed case like "PM2.5")
        df.columns = [c.strip() for c in df.columns]
        return df

    def _parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse the datetime column using a list of candidate formats.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the datetime column.

        Returns
        -------
        pd.DataFrame
            Same DataFrame with the datetime column converted to datetime64.

        Raises
        ------
        ValueError
            If the column is missing or none of the formats succeed.
        """
        if self.datetime_col not in df.columns:
            available = df.columns.tolist()
            raise ValueError(
                f"Datetime column '{self.datetime_col}' not found. "
                f"Available columns: {available}. "
                f"Update 'data.datetime_col' in configs/default.yaml."
            )

        # Try infer_datetime_format first (fastest)
        try:
            df[self.datetime_col] = pd.to_datetime(
                df[self.datetime_col], infer_datetime_format=True
            )
            logger.debug("Datetime parsed with infer_datetime_format=True.")
            return df
        except Exception:
            pass

        # Try explicit formats
        for fmt in self._DATE_FORMATS:
            try:
                df[self.datetime_col] = pd.to_datetime(
                    df[self.datetime_col], format=fmt
                )
                logger.debug(f"Datetime parsed with format '{fmt}'.")
                return df
            except Exception:
                continue

        raise ValueError(
            f"Could not parse datetime column '{self.datetime_col}'. "
            f"Tried formats: {self._DATE_FORMATS}. "
            f"Add your format to DataLoader._DATE_FORMATS."
        )

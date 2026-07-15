"""
src/pdf_ingestion/pdf_to_csv_pipeline.py
==========================================
Orchestrates the full PDF → CSV extraction pipeline.

Pipeline sequence
-----------------
    PDFReader         → raw page content
    TableExtractor    → flat, deduped, width-normalised rows
    SchemaMapper      → canonical DataFrame
    (merge sources)   → combined multi-station DataFrame
    save_csv()        → data/raw/<station_id>_raw.csv

Key design decisions
--------------------
- **One CSV per station**: each station's data is saved separately so
  the preprocessing pipeline can ingest it with the standard DataLoader.
- **Sector-62 context injection**: the sector_62 data arrives via PDF
  that may not be on disk (uploaded via chat context). A
  ``inject_context_data()`` method accepts the data as a DataFrame
  directly, bypassing PDF extraction.
- **Overlap handling**: when Sector-1 whole-year data overlaps with the
  Sector-1 May-June data, the whole-year rows take priority (they are
  already deduplicated within that source).

Usage (programmatic)
---------------------
    from src.pdf_ingestion.pdf_to_csv_pipeline import PDFtoCsvPipeline

    pipeline = PDFtoCsvPipeline(output_dir="data/raw/")
    pipeline.add_pdf("data/raw/annual_aqi.pdf",  station_id="noida_sector_1")
    pipeline.add_pdf("data/raw/may_june.pdf",    station_id="noida_sector_1")
    pipeline.add_context_dataframe(df_sector62,  station_id="noida_sector_62")
    results = pipeline.run()

Usage (one-liner)
-----------------
    run_pdf_pipeline(config)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.pdf_ingestion.pdf_reader import PDFReader
from src.pdf_ingestion.table_extractor import TableExtractor
from src.pdf_ingestion.schema_mapper import SchemaMapper
from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PDFExtractionResult:
    """Structured result from one PDF source."""
    source_name: str
    station_id: str
    pdf_type: str
    rows_extracted: int
    date_range_start: Optional[str]
    date_range_end:   Optional[str]
    columns: List[str]
    missing_counts: Dict[str, int]
    output_csv: Optional[str]
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-source entry descriptor
# ---------------------------------------------------------------------------

@dataclass
class PDFSource:
    """Describes one input source (either a PDF file or a pre-built DataFrame)."""
    station_id: str
    source_name: str
    pdf_path: Optional[Path] = None
    dataframe: Optional[pd.DataFrame] = None   # used for context-injected data


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PDFtoCsvPipeline:
    """
    Orchestrates extraction from multiple PDF sources into per-station CSVs.

    Parameters
    ----------
    output_dir : str | Path
        Directory where extracted CSV files are written.
    """

    def __init__(self, output_dir: str | Path = "data/raw/") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sources: List[PDFSource] = []
        self._extractor = TableExtractor()
        self._mapper = SchemaMapper()

    # ------------------------------------------------------------------
    # Source registration
    # ------------------------------------------------------------------

    def add_pdf(
        self,
        pdf_path: str | Path,
        station_id: str,
        source_name: Optional[str] = None,
    ) -> "PDFtoCsvPipeline":
        """
        Register a PDF file as an input source.

        Parameters
        ----------
        pdf_path : str | Path
            Path to the PDF file.
        station_id : str
            Station identifier for this PDF's data.
        source_name : str, optional
            Human-readable label. Defaults to the PDF filename.

        Returns
        -------
        self (for method chaining)
        """
        path = Path(pdf_path)
        name = source_name or path.name
        self._sources.append(PDFSource(
            station_id=station_id,
            source_name=name,
            pdf_path=path,
        ))
        logger.info(f"Registered PDF source: {name} → station={station_id}")
        return self

    def add_context_dataframe(
        self,
        df: pd.DataFrame,
        station_id: str,
        source_name: str = "context_data",
    ) -> "PDFtoCsvPipeline":
        """
        Register a pre-built DataFrame as an input source.

        Used when PDF data was parsed from the chat context (e.g., sector_62).

        Parameters
        ----------
        df : pd.DataFrame
        station_id : str
        source_name : str

        Returns
        -------
        self
        """
        self._sources.append(PDFSource(
            station_id=station_id,
            source_name=source_name,
            dataframe=df.copy(),
        ))
        logger.info(f"Registered context DataFrame: {source_name} → station={station_id}")
        return self

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def run(
        self,
        merge_same_station: bool = True,
        save_outputs: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute extraction for all registered sources.

        Parameters
        ----------
        merge_same_station : bool
            If True, DataFrames from the same ``station_id`` are merged
            (union; duplicates by date are dropped, keeping the first).
        save_outputs : bool
            If True, write one CSV per station to ``output_dir``.

        Returns
        -------
        dict
            ``{station_id: pd.DataFrame}`` — one entry per station.
        """
        logger.info("=" * 60)
        logger.info("PDF-to-CSV PIPELINE")
        logger.info(f"Sources: {len(self._sources)} | Output: {self.output_dir}")
        logger.info("=" * 60)

        # Extract all sources into DataFrames keyed by station
        station_frames: Dict[str, List[pd.DataFrame]] = {}

        for source in self._sources:
            sid = source.station_id
            try:
                df = self._process_source(source)
                station_frames.setdefault(sid, []).append(df)
            except Exception as exc:
                logger.error(
                    f"Source '{source.source_name}' failed: {exc}", exc_info=True
                )

        # Merge same-station frames
        merged: Dict[str, pd.DataFrame] = {}
        for sid, frames in station_frames.items():
            if merge_same_station and len(frames) > 1:
                df_merged = self._merge_station_frames(frames, sid)
            else:
                df_merged = frames[0]
            merged[sid] = df_merged
            logger.info(
                f"Station '{sid}': {len(df_merged):,} rows | "
                f"cols={df_merged.columns.tolist()}"
            )

        # Save outputs
        if save_outputs:
            for sid, df in merged.items():
                self._save_station_csv(df, sid)

        logger.info("=" * 60)
        logger.info(
            f"Pipeline complete | "
            f"{sum(len(d) for d in merged.values())} total rows across "
            f"{len(merged)} station(s)"
        )
        return merged

    # ------------------------------------------------------------------
    # Per-source processing
    # ------------------------------------------------------------------

    def _process_source(self, source: PDFSource) -> pd.DataFrame:
        """
        Process one source (PDF file or context DataFrame) into a
        canonically-named DataFrame.
        """
        sid = source.station_id

        # Context DataFrame path (sector_62 data injected from chat)
        if source.dataframe is not None:
            logger.info(
                f"[{sid}] Processing context DataFrame: {source.source_name} | "
                f"shape={source.dataframe.shape}"
            )
            return self._normalise_context_df(source.dataframe, sid)

        # PDF path
        logger.info(f"[{sid}] Processing PDF: {source.source_name}")
        reader = PDFReader(source.pdf_path)
        raw_pages = reader.extract_all_pages()

        rows, pdf_type = self._extractor.merge_and_clean(
            raw_pages, source_name=source.source_name
        )

        df = self._mapper.map(rows, pdf_type=pdf_type, station_id=sid)
        logger.info(
            f"[{sid}] '{source.source_name}' extracted | "
            f"type={pdf_type} | rows={len(df)}"
        )
        return df

    def _normalise_context_df(
        self, df: pd.DataFrame, station_id: str
    ) -> pd.DataFrame:
        """
        Normalise a DataFrame that was built directly from context (not from PDF).

        Applies alias renaming, dtype coercion, and station column attachment.
        """
        from src.preprocessing.validator import COLUMN_ALIASES
        df = df.copy()

        # Rename aliases
        rename_map = {c: COLUMN_ALIASES[c] for c in df.columns if c in COLUMN_ALIASES}
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.debug(f"[{station_id}] Context aliases renamed: {rename_map}")

        # Ensure station column
        if "station" not in df.columns:
            df.insert(1, "station", station_id)

        # Ensure date column is datetime
        date_col = next((c for c in ["date", "Date", "datetime"] if c in df.columns), None)
        if date_col and date_col != "date":
            df = df.rename(columns={date_col: "date"})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).reset_index(drop=True)

        # Coerce numerics
        numeric_canonical = {
            "aqi","pm25","pm10","no2","so2","co","o3","nh3","pb",
            "temperature","humidity","wind_speed","wind_direction","pressure",
        }
        for col in df.columns:
            if col in numeric_canonical:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _merge_station_frames(
        self,
        frames: List[pd.DataFrame],
        station_id: str,
    ) -> pd.DataFrame:
        """
        Merge multiple DataFrames for the same station.

        Strategy: concatenate, sort by date, drop duplicate dates
        (keeping the first occurrence — the order in which sources were
        registered determines priority).
        """
        merged = pd.concat(frames, axis=0, ignore_index=True)
        before = len(merged)

        if "date" in merged.columns:
            merged = merged.sort_values("date").reset_index(drop=True)
            merged = merged.drop_duplicates(subset=["date"], keep="first")

        after = len(merged)
        if before != after:
            logger.info(
                f"[{station_id}] Merged {len(frames)} sources: "
                f"{before} → {after} rows ({before - after} duplicates removed)"
            )

        # Align all column dtypes after merge
        return merged.reset_index(drop=True)

    def _save_station_csv(self, df: pd.DataFrame, station_id: str) -> Path:
        """Save a station DataFrame to CSV in the output directory."""
        dest = self.output_dir / f"{station_id}_raw.csv"
        df.to_csv(dest, index=False)
        logger.info(f"[{station_id}] Saved → {dest} | rows={len(df):,}")
        return dest

    # ------------------------------------------------------------------
    # Validation report
    # ------------------------------------------------------------------

    def generate_extraction_report(
        self, station_dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, dict]:
        """
        Generate a structured validation report for all extracted DataFrames.

        Parameters
        ----------
        station_dfs : dict
            Output of ``run()``.

        Returns
        -------
        dict
            ``{station_id: {rows, cols, date_range, missing, duplicates, …}}``
        """
        report = {}
        for sid, df in station_dfs.items():
            date_min = str(df["date"].min().date()) if "date" in df.columns else "N/A"
            date_max = str(df["date"].max().date()) if "date" in df.columns else "N/A"

            missing = {
                col: int(df[col].isna().sum())
                for col in df.columns
                if df[col].isna().sum() > 0
            }
            dup_count = int(df.duplicated(subset=["date"]).sum()) if "date" in df.columns else 0

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            stats = {}
            for col in numeric_cols:
                s = df[col].dropna()
                if len(s):
                    stats[col] = {
                        "min": round(float(s.min()), 2),
                        "max": round(float(s.max()), 2),
                        "mean": round(float(s.mean()), 2),
                        "null_count": int(df[col].isna().sum()),
                    }

            report[sid] = {
                "rows": len(df),
                "columns": df.columns.tolist(),
                "n_columns": len(df.columns),
                "date_range_start": date_min,
                "date_range_end": date_max,
                "missing_value_counts": missing,
                "duplicate_date_rows": dup_count,
                "numeric_stats": stats,
            }

        return report


# ---------------------------------------------------------------------------
# Functional entry point
# ---------------------------------------------------------------------------

def run_pdf_pipeline(
    config: AppConfig,
    pdf_sources: Optional[List[Tuple[str, str]]] = None,
    context_dataframes: Optional[Dict[str, pd.DataFrame]] = None,
    save_outputs: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function called by ``main.py --stage pdf_extract``.

    Parameters
    ----------
    config : AppConfig
    pdf_sources : list of (pdf_path, station_id), optional
        If None, scans ``config.paths.raw_data`` for all ``.pdf`` files.
    context_dataframes : dict of {station_id: DataFrame}, optional
        Pre-built DataFrames injected from non-file sources (chat context).
    save_outputs : bool

    Returns
    -------
    dict of {station_id: DataFrame}
    """
    pipeline = PDFtoCsvPipeline(output_dir=config.paths.raw_data)

    # Register PDF files
    if pdf_sources:
        for pdf_path, station_id in pdf_sources:
            pipeline.add_pdf(pdf_path, station_id=station_id)
    else:
        # Auto-discover all PDFs in raw data dir
        raw_dir = Path(config.paths.raw_data)
        for pdf_file in sorted(raw_dir.glob("*.pdf")):
            sid = config.project.station_id
            pipeline.add_pdf(pdf_file, station_id=sid)
            logger.info(f"Auto-discovered PDF: {pdf_file.name}")

    # Register context DataFrames
    if context_dataframes:
        for station_id, df in context_dataframes.items():
            pipeline.add_context_dataframe(df, station_id=station_id)

    results = pipeline.run(save_outputs=save_outputs)

    # Print extraction report
    report = pipeline.generate_extraction_report(results)
    _print_report(report)

    return results


def _print_report(report: Dict[str, dict]) -> None:
    """Pretty-print the extraction validation report to stdout."""
    print("\n" + "=" * 65)
    print("  PDF EXTRACTION VALIDATION REPORT")
    print("=" * 65)
    for sid, info in report.items():
        print(f"\n  Station  : {sid}")
        print(f"  Rows     : {info['rows']:,}")
        print(f"  Columns  : {info['n_columns']}  {info['columns']}")
        print(f"  Date range: {info['date_range_start']} → {info['date_range_end']}")
        if info["missing_value_counts"]:
            print(f"  Missing  : {info['missing_value_counts']}")
        else:
            print("  Missing  : None ✅")
        if info["duplicate_date_rows"]:
            print(f"  Duplicates: {info['duplicate_date_rows']} ⚠️")
        else:
            print("  Duplicates: None ✅")
        print("\n  Numeric column stats:")
        print(f"  {'Column':<15} {'Min':>8} {'Max':>8} {'Mean':>8} {'Nulls':>6}")
        print(f"  {'-'*50}")
        for col, s in info["numeric_stats"].items():
            print(
                f"  {col:<15} {s['min']:>8.1f} {s['max']:>8.1f} "
                f"{s['mean']:>8.1f} {s['null_count']:>6}"
            )
    print("\n" + "=" * 65)

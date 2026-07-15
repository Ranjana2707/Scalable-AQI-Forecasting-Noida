"""
src/preprocessing/pipeline.py
================================
Orchestrates the complete Phase 2 preprocessing pipeline.

Pipeline sequence
-----------------
    DataLoader
        ↓  raw DataFrame (datetime col, station col)
    SchemaValidator
        ↓  renamed, typed, optional cols added as NaN
    DataCleaner
        ↓  DatetimeIndex, hourly grid, imputed, non-negative
    OutlierHandler
        ↓  outliers capped within physical bounds
    QualityReporter
        ↓  JSON + HTML report written to outputs/reports/
    save_processed()
        ↓  CSV written to data/processed/<station_id>_clean.csv
    run_split()
        ↓  train / val / test CSVs in data/processed/

Entry points
------------
- ``run_preprocessing_pipeline(config)``   — called by ``main.py``
- ``PreprocessingPipeline(config)``        — class API for notebooks / tests

Usage (CLI)
-----------
    python main.py --stage preprocess --config configs/default.yaml

Usage (programmatic)
--------------------
    from src.preprocessing.pipeline import run_preprocessing_pipeline
    cfg = load_config("configs/default.yaml")
    result = run_preprocessing_pipeline(cfg)
    df_train = result.train
    df_val   = result.val
    df_test  = result.test
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.loader import DataLoader
from src.preprocessing.outlier_handler import OutlierHandler
from src.preprocessing.quality_report import QualityReporter
from src.preprocessing.validator import SchemaValidator
from src.utils.config import AppConfig
from src.utils.logger import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline result container
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingResult:
    """
    Structured result returned by the preprocessing pipeline.

    Attributes
    ----------
    raw : pd.DataFrame
        DataFrame as loaded from disk (before any transformation).
    clean : pd.DataFrame
        Fully preprocessed DataFrame (entire dataset, no split).
    train : pd.DataFrame
        Training split (chronologically first).
    val : pd.DataFrame
        Validation split (middle, with gap before test).
    test : pd.DataFrame
        Test split (chronologically last, unseen during training).
    station_id : str
        Station identifier.
    elapsed_seconds : float
        Total wall-clock time for the pipeline run.
    """
    raw: pd.DataFrame
    clean: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    station_id: str
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Class-based pipeline (reusable in notebooks)
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """
    Encapsulates the full preprocessing pipeline as a reusable object.

    Using a class (rather than a plain function) lets notebooks and tests
    inspect intermediate states (e.g., ``pipeline.cleaner.column_medians``)
    and reuse fitted transformers for inference.

    Parameters
    ----------
    config : AppConfig
        Loaded project configuration.

    Attributes
    ----------
    loader    : DataLoader
    validator : SchemaValidator
    cleaner   : DataCleaner
    outlier_handler : OutlierHandler
    reporter  : QualityReporter
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.loader = DataLoader(config)
        self.validator = SchemaValidator(config)
        self.cleaner = DataCleaner(config)
        self.outlier_handler = OutlierHandler(config)
        self.reporter = QualityReporter(config)

    def run(
        self,
        station_id: Optional[str] = None,
        save_outputs: bool = True,
        print_report: bool = True,
    ) -> PreprocessingResult:
        """
        Execute the full preprocessing pipeline for one station.

        Parameters
        ----------
        station_id : str, optional
            Overrides ``config.project.station_id``.  Pass explicitly for
            multi-station loops.
        save_outputs : bool
            Write processed CSVs and quality reports to disk.
        print_report : bool
            Print quality report summary to stdout.

        Returns
        -------
        PreprocessingResult
        """
        sid = station_id or self.config.project.station_id
        t_start = time.perf_counter()

        logger.info("=" * 65)
        logger.info(f"PREPROCESSING PIPELINE — Station: {sid}")
        logger.info("=" * 65)

        # ── Step 1: Load ──────────────────────────────────────────────
        logger.info("[1/6] Loading raw data...")
        df_raw = self.loader.load(station_id=sid)
        logger.info(f"      Raw shape: {df_raw.shape}")

        # ── Step 2: Schema validation ─────────────────────────────────
        logger.info("[2/6] Validating schema...")
        df_validated = self.validator.validate(df_raw, station_id=sid)
        val_report = self.validator.get_validation_report()

        # ── Step 3: Clean ─────────────────────────────────────────────
        logger.info("[3/6] Cleaning data...")
        df_cleaned = self.cleaner.fit_transform(df_validated, station_id=sid)

        # ── Step 4: Outlier handling ──────────────────────────────────
        logger.info("[4/6] Handling outliers...")
        df_clean = self.outlier_handler.fit_transform(df_cleaned, station_id=sid)

        # ── Step 5: Quality report ────────────────────────────────────
        logger.info("[5/6] Generating quality report...")
        outlier_df = self.outlier_handler.get_outlier_summary()
        quality_rpt = self.reporter.generate(
            df_raw=df_raw,
            df_clean=df_clean,
            station_id=sid,
            validation_report=val_report,
            outlier_summary_df=outlier_df,
        )
        if print_report:
            quality_rpt.print_summary()
        if save_outputs:
            self.reporter.save(quality_rpt, station_id=sid)

        # ── Step 6: Save & split ──────────────────────────────────────
        logger.info("[6/6] Saving processed data and splitting...")
        if save_outputs:
            self._save_clean(df_clean, sid)

        train_df, val_df, test_df = _time_series_split(df_clean, self.config, sid)

        if save_outputs:
            self._save_splits(train_df, val_df, test_df, sid)

        elapsed = time.perf_counter() - t_start
        logger.info(
            f"Pipeline complete | station={sid} | elapsed={elapsed:.2f}s | "
            f"train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}"
        )
        logger.info("=" * 65)

        return PreprocessingResult(
            raw=df_raw,
            clean=df_clean,
            train=train_df,
            val=val_df,
            test=test_df,
            station_id=sid,
            elapsed_seconds=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_clean(self, df: pd.DataFrame, station_id: str) -> None:
        dest = Path(self.config.paths.processed_data) / f"{station_id}_clean.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest)
        logger.info(f"Clean data saved → {dest}")

    def _save_splits(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        station_id: str,
    ) -> None:
        proc_dir = Path(self.config.paths.processed_data)
        proc_dir.mkdir(parents=True, exist_ok=True)
        for name, split_df in [("train", train), ("val", val), ("test", test)]:
            path = proc_dir / f"{station_id}_{name}.csv"
            split_df.to_csv(path)
            logger.info(f"{name.capitalize()} split saved → {path} | rows={len(split_df):,}")


# ---------------------------------------------------------------------------
# Standalone split function
# ---------------------------------------------------------------------------

def _time_series_split(
    df: pd.DataFrame,
    config: AppConfig,
    station_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply a chronological train / val / test split with a gap.

    Timeline diagram::

        |<────── train ──────>|< gap >|<── val ──>|< gap >|<── test ──>|
        0                    t1      t2           t3      t4            N

    The gap (``config.split.gap_hours`` hours) between sets prevents
    any information from a recent observation leaking into the next split
    through autocorrelation in lag features.

    Parameters
    ----------
    df : pd.DataFrame
        Fully preprocessed DataFrame with a DatetimeIndex.
    config : AppConfig
    station_id : str
        For logging.

    Returns
    -------
    tuple of (train_df, val_df, test_df)

    Raises
    ------
    ValueError
        If the dataset is too small for the configured split sizes.
    """
    n = len(df)
    test_size = config.split.test_size
    val_size = config.split.val_size
    gap = config.split.gap_hours

    test_n = int(n * test_size)
    val_n = int(n * val_size)

    test_start = n - test_n
    val_end = test_start - gap
    val_start = val_end - val_n
    train_end = val_start - gap

    if train_end <= 0:
        raise ValueError(
            f"[{station_id}] Dataset too small for configured split. "
            f"n={n}, train_end={train_end}. "
            "Reduce test_size / val_size or gap_hours in configs/default.yaml."
        )

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[val_start:val_end].copy()
    test_df = df.iloc[test_start:].copy()

    logger.info(
        f"[{station_id}] Split | "
        f"train={len(train_df):,} ({train_df.index.min()} → {train_df.index.max()}) | "
        f"val={len(val_df):,} | "
        f"test={len(test_df):,} ({test_df.index.min()} → {test_df.index.max()}) | "
        f"gap={gap}h"
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Functional entry point (called by main.py)
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(
    config: AppConfig,
    station_id: Optional[str] = None,
    save_outputs: bool = True,
    print_report: bool = True,
) -> PreprocessingResult:
    """
    Convenience function that instantiates ``PreprocessingPipeline`` and
    calls ``run()``.  This is the single function called by ``main.py``.

    Parameters
    ----------
    config : AppConfig
        Loaded project configuration.
    station_id : str, optional
        Override for multi-station loops.
    save_outputs : bool
    print_report : bool

    Returns
    -------
    PreprocessingResult
    """
    set_global_seed(config.project.seed)
    pipeline = PreprocessingPipeline(config)
    return pipeline.run(
        station_id=station_id,
        save_outputs=save_outputs,
        print_report=print_report,
    )

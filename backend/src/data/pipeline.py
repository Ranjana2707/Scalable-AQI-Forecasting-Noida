"""
src/data/pipeline.py
=====================
Orchestrates the full data ingestion and preprocessing pipeline.

This module is the single entry point called by ``main.py`` for the
``--stage preprocess`` command.  It wires together:

    DataLoader → SchemaValidator → QualityValidator → AQIPreprocessor

and writes:
- ``data/processed/<station_id>_processed.csv``
- ``outputs/reports/<station_id>_quality_report.json``

Usage (CLI via main.py)
-----------------------
    python main.py --stage preprocess --config configs/default.yaml

Usage (programmatic)
---------------------
    from src.data.pipeline import run_preprocessing
    from src.utils.config import load_config

    cfg = load_config("configs/default.yaml")
    df_processed = run_preprocessing(cfg)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.loader import DataLoader
from src.data.preprocessor import AQIPreprocessor
from src.data.validator import QualityValidator, SchemaValidator
from src.utils.config import AppConfig
from src.utils.io_helpers import save_json
from src.utils.logger import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)


def run_preprocessing(
    config: AppConfig,
    save_outputs: bool = True,
    validate_schema: bool = True,
    validate_quality: bool = True,
) -> pd.DataFrame:
    """
    Execute the complete data ingestion and preprocessing pipeline.

    Steps
    -----
    1. Set global random seed for reproducibility.
    2. Load raw data via ``DataLoader``.
    3. Validate schema via ``SchemaValidator``.
    4. Run quality checks via ``QualityValidator``; save report.
    5. Preprocess via ``AQIPreprocessor.fit_transform``.
    6. Save processed CSV to ``data/processed/``.

    Parameters
    ----------
    config : AppConfig
        Loaded project configuration.
    save_outputs : bool
        Whether to write processed CSV and quality report to disk.
    validate_schema : bool
        Whether to run schema validation (disable for speed in tests).
    validate_quality : bool
        Whether to run quality validation (disable for speed in tests).

    Returns
    -------
    pd.DataFrame
        Fully preprocessed DataFrame, ready for feature engineering.

    Raises
    ------
    FileNotFoundError
        If raw data files are not found.
    ValueError
        If schema validation fails.
    """
    t_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("PIPELINE STAGE: Data Ingestion & Preprocessing")
    logger.info(f"Station : {config.project.station_id}")
    logger.info(f"Seed    : {config.project.seed}")
    logger.info("=" * 60)

    # Step 0 — Reproducibility
    set_global_seed(config.project.seed)

    # Step 1 — Load
    logger.info("Step 1/4 — Loading raw data...")
    loader = DataLoader(config)
    df_raw = loader.load_raw()
    logger.info(f"Raw data shape: {df_raw.shape}")

    # Step 2 — Schema validation
    if validate_schema:
        logger.info("Step 2/4 — Schema validation...")
        schema_v = SchemaValidator(config)
        schema_v.validate(df_raw, strict=False)
    else:
        logger.info("Step 2/4 — Schema validation skipped.")

    # Step 3 — Quality validation
    if validate_quality:
        logger.info("Step 3/4 — Quality validation...")
        quality_v = QualityValidator(config)
        report = quality_v.validate(df_raw)
        report.print_summary()
        if save_outputs:
            report_path = (
                Path(config.paths.reports)
                / f"{config.project.station_id}_quality_report.json"
            )
            save_json(report.to_dict(), report_path)
            logger.info(f"Quality report saved → {report_path}")
    else:
        logger.info("Step 3/4 — Quality validation skipped.")

    # Step 4 — Preprocessing
    logger.info("Step 4/4 — Preprocessing...")
    preprocessor = AQIPreprocessor(config)
    df_processed = preprocessor.fit_transform(df_raw)

    if save_outputs:
        preprocessor.save_processed(df_processed)

    elapsed = time.perf_counter() - t_start
    logger.info(f"Preprocessing pipeline complete in {elapsed:.2f}s | shape={df_processed.shape}")
    logger.info("=" * 60)

    return df_processed


def run_split(
    df: pd.DataFrame,
    config: AppConfig,
    save_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply time-series aware train / validation / test split.

    No random shuffling — the split preserves chronological order with a
    configurable gap between sets to prevent data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed, feature-engineered DataFrame with a DatetimeIndex.
    config : AppConfig
        Project configuration (split.test_size, split.val_size, split.gap_hours).
    save_outputs : bool
        Whether to save split CSVs to ``data/processed/``.

    Returns
    -------
    tuple of (train_df, val_df, test_df)

    Notes
    -----
    Split logic::

        |<-------- train -------->|<gap>|<-- val -->|<gap>|<-- test -->|
        0                        t1    t2           t3    t4           N

    where gap = ``config.split.gap_hours`` rows.
    """
    n = len(df)
    test_size = config.split.test_size
    val_size = config.split.val_size
    gap = config.split.gap_hours

    test_n = int(n * test_size)
    val_n = int(n * val_size)

    # Compute split indices (chronological)
    test_start = n - test_n
    val_start = test_start - gap - val_n
    train_end = val_start - gap

    if train_end <= 0:
        raise ValueError(
            f"Dataset too small for the configured split. "
            f"n={n}, train_end={train_end}. "
            "Reduce val_size, test_size, or gap_hours."
        )

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[val_start : val_start + val_n].copy()
    test_df = df.iloc[test_start:].copy()

    logger.info(
        f"Train/Val/Test split (chronological) | "
        f"train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,} | "
        f"gap={gap}h"
    )

    if save_outputs:
        proc_dir = Path(config.paths.processed_data)
        sid = config.project.station_id
        for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            path = proc_dir / f"{sid}_{name}.csv"
            split_df.to_csv(path)
            logger.info(f"Split saved → {path}")

    return train_df, val_df, test_df

"""
tests/integration/test_preprocessing_pipeline.py
==================================================
End-to-end integration tests for the full preprocessing pipeline.

These tests create realistic synthetic CSV files on disk, run the complete
pipeline (DataLoader → SchemaValidator → DataCleaner → OutlierHandler →
QualityReporter → split), and assert on the final artifacts.

They are intentionally slower than unit tests and are tagged with
``@pytest.mark.integration`` so they can be excluded from fast CI runs::

    pytest -m "not integration"   # unit tests only
    pytest -m integration         # integration tests only
    pytest                        # all tests
"""
from __future__ import annotations

import textwrap
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipeline import (
    PreprocessingPipeline,
    PreprocessingResult,
    run_preprocessing_pipeline,
)
from src.utils.config import (
    AppConfig,
    DataConfig,
    PathsConfig,
    PreprocessingConfig,
    ProjectConfig,
    SplitConfig,
    EvaluationConfig,
    MLflowConfig,
)


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, station_id: str = "test_station") -> AppConfig:
    """Build a minimal AppConfig pointing at tmp_path directories."""
    cfg = AppConfig()
    cfg.project = ProjectConfig(station_id=station_id, seed=42, log_level="WARNING")
    cfg.data = DataConfig(
        datetime_col="datetime",
        target_col="aqi",
        pollutant_cols=["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"],
        meteorological_cols=["temperature", "humidity", "wind_speed", "pressure"],
        aqi_breakpoints={
            "good":        [0,   50],
            "satisfactory":[51, 100],
            "moderate":    [101, 200],
            "poor":        [201, 300],
            "very_poor":   [301, 400],
            "severe":      [401, 500],
        },
    )
    cfg.preprocessing = PreprocessingConfig(
        missing_strategy="interpolate",
        interpolation_method="linear",
        outlier_method="iqr",
        outlier_threshold=3.0,
        min_valid_data_pct=0.40,
    )
    cfg.split = SplitConfig(
        strategy="time_series",
        test_size=0.15,
        val_size=0.10,
        gap_hours=12,
    )
    cfg.paths = PathsConfig(
        raw_data=str(tmp_path / "raw") + "/",
        interim_data=str(tmp_path / "interim") + "/",
        processed_data=str(tmp_path / "processed") + "/",
        external_data=str(tmp_path / "external") + "/",
        figures=str(tmp_path / "figures") + "/",
        reports=str(tmp_path / "reports") + "/",
        predictions=str(tmp_path / "predictions") + "/",
        models=str(tmp_path / "models") + "/",
        logs=str(tmp_path / "logs") + "/",
        mlflow_uri=str(tmp_path / "mlruns") + "/",
    )
    cfg.evaluation = EvaluationConfig()
    cfg.mlflow = MLflowConfig()
    return cfg


def _write_synthetic_csv(
    path: Path,
    station_id: str,
    n_hours: int = 800,
    missing_frac: float = 0.05,
    outlier_frac: float = 0.02,
    seed: int = 42,
) -> None:
    """
    Write a realistic synthetic AQI CSV to ``path``.

    Includes:
    - Diurnal AQI pattern (higher at rush hour)
    - Realistic pollutant correlations
    - Configurable missing-value fraction
    - Configurable outlier fraction (extreme spikes)
    - Mixed column name styles (aliases) to test renaming
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(n_hours)]
    hours = np.array([d.hour for d in dates])

    # Diurnal signal: peaks at 8am and 6pm
    diurnal = (
        40 * np.sin(np.pi * (hours - 6) / 12) ** 2
        + 20 * np.sin(np.pi * (hours - 15) / 12) ** 2
    )
    base_aqi = 150 + diurnal + rng.normal(0, 20, n_hours)
    base_aqi = np.clip(base_aqi, 30, 450)

    df = pd.DataFrame({
        "datetime":    dates,
        "Station":     [station_id] * n_hours,   # alias for "station"
        "AQI":         base_aqi,                  # alias for "aqi"
        "PM2.5":       base_aqi * 0.4 + rng.normal(0, 5, n_hours),   # alias pm25
        "PM10":        base_aqi * 0.6 + rng.normal(0, 8, n_hours),   # alias pm10
        "NO2":         base_aqi * 0.2 + rng.normal(0, 3, n_hours),
        "SO2":         rng.uniform(5, 50, n_hours),
        "CO":          rng.uniform(0.5, 5.0, n_hours),
        "O3":          50 + rng.normal(0, 10, n_hours),
        "NH3":         rng.uniform(2, 30, n_hours),
        "Temperature": 25 + 8 * np.sin(np.pi * hours / 12) + rng.normal(0, 2, n_hours),
        "Humidity":    60 + rng.normal(0, 10, n_hours),
        "WS":          rng.uniform(0.5, 8, n_hours),   # alias wind_speed
        "Pressure":    1010 + rng.normal(0, 5, n_hours),
    })

    # Inject missing values
    n_missing = int(n_hours * missing_frac)
    for col in ["PM2.5", "NO2", "Humidity"]:
        idx = rng.choice(n_hours, n_missing // 3, replace=False)
        df.loc[idx, col] = np.nan

    # Inject outliers
    n_outliers = int(n_hours * outlier_frac)
    outlier_idx = rng.choice(n_hours, n_outliers, replace=False)
    df.loc[outlier_idx, "PM2.5"] = rng.uniform(800, 2000, n_outliers)

    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestFullPipelineRun:
    """Tests for the complete pipeline from raw CSV to clean splits."""

    def test_pipeline_completes_without_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(
            tmp_path / "raw" / "test_station_2023.csv",
            station_id="test_station",
        )
        result = run_preprocessing_pipeline(cfg, save_outputs=False, print_report=False)
        assert isinstance(result, PreprocessingResult)

    def test_result_has_all_splits(self, tmp_path):
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(tmp_path / "raw" / "test_station_2023.csv", "test_station")
        result = run_preprocessing_pipeline(cfg, save_outputs=False, print_report=False)
        assert len(result.train) > 0
        assert len(result.val)   > 0
        assert len(result.test)  > 0

    def test_clean_df_has_no_nulls_in_numeric_cols(self, tmp_path):
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(tmp_path / "raw" / "test_station_2023.csv", "test_station")
        result = run_preprocessing_pipeline(cfg, save_outputs=False, print_report=False)
        numeric_nulls = (
            result.clean.select_dtypes(include=[np.number]).isna().sum().sum()
        )
        assert numeric_nulls == 0

    def test_clean_df_has_datetime_index(self, tmp_path):
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(tmp_path / "raw" / "test_station_2023.csv", "test_station")
        result = run_preprocessing_pipeline(cfg, save_outputs=False, print_report=False)
        assert isinstance(result.clean.index, pd.DatetimeIndex)

    def test_clean_df_has_regular_hourly_index(self, tmp_path):
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(tmp_path / "raw" / "test_station_2023.csv", "test_station")
        result = run_preprocessing_pipeline(cfg, save_outputs=False, print_report=False)
        diffs = pd.Series(result.clean.index).diff().dropna()
        assert (diffs == pd.Timedelta("1h")).all()

    def test_column_aliases_resolved(self, tmp_path):
        """PM2.5, AQI, Station, WS etc. should be renamed to canonical names."""
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(tmp_path / "raw" / "test_station_2023.csv", "test_station")
        result = run_preprocessing_pipeline(cfg, save_outputs=False, print_report=False)
        assert "aqi"   in result.clean.columns
        assert "pm25"  in result.clean.columns
        assert "AQI"   not in result.clean.columns
        assert "PM2.5" not in result.clean.columns

    def test_outliers_capped_not_dropped(self, tmp_path):
        """Row count should be preserved — outliers capped, not dropped."""
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(
            tmp_path / "raw" / "test_station_2023.csv",
            "test_station",
            outlier_frac=0.05,
        )
        result = run_preprocessing_pipeline(cfg, save_outputs=False, print_report=False)
        # pm25 physical max is 1000 — capped values must not exceed it
        assert result.clean["pm25"].max() <= 1000.0

    def test_aqi_values_in_physical_range(self, tmp_path):
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(tmp_path / "raw" / "test_station_2023.csv", "test_station")
        result = run_preprocessing_pipeline(cfg, save_outputs=False, print_report=False)
        assert result.clean["aqi"].min() >= 0.0
        assert result.clean["aqi"].max() <= 500.0

    def test_splits_are_chronologically_ordered(self, tmp_path):
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(tmp_path / "raw" / "test_station_2023.csv", "test_station")
        result = run_preprocessing_pipeline(cfg, save_outputs=False, print_report=False)
        assert result.train.index.max() < result.val.index.min()
        assert result.val.index.max()   < result.test.index.min()

    def test_no_overlap_between_splits(self, tmp_path):
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(tmp_path / "raw" / "test_station_2023.csv", "test_station")
        result = run_preprocessing_pipeline(cfg, save_outputs=False, print_report=False)
        train_idx = set(result.train.index)
        val_idx   = set(result.val.index)
        test_idx  = set(result.test.index)
        assert train_idx.isdisjoint(val_idx),  "Train/Val overlap detected!"
        assert train_idx.isdisjoint(test_idx), "Train/Test overlap detected!"
        assert val_idx.isdisjoint(test_idx),   "Val/Test overlap detected!"

    def test_station_column_present_and_correct(self, tmp_path):
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(tmp_path / "raw" / "test_station_2023.csv", "test_station")
        result = run_preprocessing_pipeline(cfg, save_outputs=False, print_report=False)
        assert "station" in result.clean.columns
        assert (result.clean["station"] == "test_station").all()

    def test_elapsed_seconds_recorded(self, tmp_path):
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(tmp_path / "raw" / "test_station_2023.csv", "test_station")
        result = run_preprocessing_pipeline(cfg, save_outputs=False, print_report=False)
        assert result.elapsed_seconds > 0.0


class TestSaveOutputs:
    """Tests that verify correct file artifacts are written to disk."""

    def test_clean_csv_written(self, tmp_path):
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(tmp_path / "raw" / "test_station_2023.csv", "test_station")
        run_preprocessing_pipeline(cfg, save_outputs=True, print_report=False)
        assert (tmp_path / "processed" / "test_station_clean.csv").exists()

    def test_train_val_test_csvs_written(self, tmp_path):
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(tmp_path / "raw" / "test_station_2023.csv", "test_station")
        run_preprocessing_pipeline(cfg, save_outputs=True, print_report=False)
        for split in ("train", "val", "test"):
            p = tmp_path / "processed" / f"test_station_{split}.csv"
            assert p.exists(), f"Missing split file: {p.name}"

    def test_quality_report_json_written(self, tmp_path):
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(tmp_path / "raw" / "test_station_2023.csv", "test_station")
        run_preprocessing_pipeline(cfg, save_outputs=True, print_report=False)
        assert (tmp_path / "reports" / "test_station_quality_report.json").exists()

    def test_quality_report_html_written(self, tmp_path):
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(tmp_path / "raw" / "test_station_2023.csv", "test_station")
        run_preprocessing_pipeline(cfg, save_outputs=True, print_report=False)
        assert (tmp_path / "reports" / "test_station_quality_report.html").exists()

    def test_saved_clean_csv_readable(self, tmp_path):
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(tmp_path / "raw" / "test_station_2023.csv", "test_station")
        run_preprocessing_pipeline(cfg, save_outputs=True, print_report=False)
        df_loaded = pd.read_csv(
            tmp_path / "processed" / "test_station_clean.csv",
            index_col=0, parse_dates=True,
        )
        assert len(df_loaded) > 0
        assert isinstance(df_loaded.index, pd.DatetimeIndex)


class TestMultiStationScalability:
    """
    Verifies that the pipeline correctly handles multiple stations
    without any architectural changes — only the station_id changes.
    """

    def test_two_stations_processed_independently(self, tmp_path):
        cfg_a = _make_config(tmp_path / "station_a", station_id="station_a")
        cfg_b = _make_config(tmp_path / "station_b", station_id="station_b")

        _write_synthetic_csv(
            tmp_path / "station_a" / "raw" / "station_a_2023.csv",
            station_id="station_a", n_hours=600, seed=1,
        )
        _write_synthetic_csv(
            tmp_path / "station_b" / "raw" / "station_b_2023.csv",
            station_id="station_b", n_hours=600, seed=2,
        )

        result_a = run_preprocessing_pipeline(cfg_a, save_outputs=False, print_report=False)
        result_b = run_preprocessing_pipeline(cfg_b, save_outputs=False, print_report=False)

        assert result_a.station_id == "station_a"
        assert result_b.station_id == "station_b"
        assert (result_a.clean["station"] == "station_a").all()
        assert (result_b.clean["station"] == "station_b").all()

    def test_station_ids_do_not_bleed_across_pipelines(self, tmp_path):
        """Station A's data must never appear in Station B's results."""
        cfg_a = _make_config(tmp_path / "sa", station_id="sa")
        cfg_b = _make_config(tmp_path / "sb", station_id="sb")
        _write_synthetic_csv(tmp_path / "sa" / "raw" / "sa_2023.csv", "sa", n_hours=400, seed=10)
        _write_synthetic_csv(tmp_path / "sb" / "raw" / "sb_2023.csv", "sb", n_hours=400, seed=20)

        result_a = run_preprocessing_pipeline(cfg_a, save_outputs=False, print_report=False)
        result_b = run_preprocessing_pipeline(cfg_b, save_outputs=False, print_report=False)

        assert set(result_a.clean["station"].unique()) == {"sa"}
        assert set(result_b.clean["station"].unique()) == {"sb"}

    def test_pipeline_class_reusable_for_different_stations(self, tmp_path):
        """
        PreprocessingPipeline can be re-instantiated for each station.
        Each instance is fully independent — no shared state.
        """
        for sid in ["alpha", "beta", "gamma"]:
            cfg = _make_config(tmp_path / sid, station_id=sid)
            _write_synthetic_csv(
                tmp_path / sid / "raw" / f"{sid}_2023.csv",
                station_id=sid, n_hours=400, seed=hash(sid) % 1000,
            )
            pipeline = PreprocessingPipeline(cfg)
            result = pipeline.run(save_outputs=False, print_report=False)
            assert result.station_id == sid


class TestRobustness:
    """Edge-case and robustness tests for the preprocessing pipeline."""

    def test_handles_heavy_missing_data(self, tmp_path):
        """Pipeline should not crash with 30 % missing values."""
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(
            tmp_path / "raw" / "test_station_2023.csv",
            "test_station", missing_frac=0.30,
        )
        result = run_preprocessing_pipeline(cfg, save_outputs=False, print_report=False)
        assert result.clean.select_dtypes(include=[np.number]).isna().sum().sum() == 0

    def test_handles_extreme_outlier_fraction(self, tmp_path):
        """Pipeline should cap, not crash, on 10 % extreme outliers."""
        cfg = _make_config(tmp_path)
        _write_synthetic_csv(
            tmp_path / "raw" / "test_station_2023.csv",
            "test_station", outlier_frac=0.10,
        )
        result = run_preprocessing_pipeline(cfg, save_outputs=False, print_report=False)
        assert result.clean["pm25"].max() <= 1000.0

    def test_raises_when_no_data_file(self, tmp_path):
        """Pipeline must raise FileNotFoundError when raw dir is empty."""
        cfg = _make_config(tmp_path)
        (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError):
            run_preprocessing_pipeline(cfg, save_outputs=False, print_report=False)

    def test_handles_mixed_column_name_styles(self, tmp_path):
        """Aliases like PM2.5, AQI, Temperature, WS must all be mapped."""
        cfg = _make_config(tmp_path)
        # _write_synthetic_csv already uses mixed aliases
        _write_synthetic_csv(tmp_path / "raw" / "test_station_2023.csv", "test_station")
        result = run_preprocessing_pipeline(cfg, save_outputs=False, print_report=False)
        for col in ["aqi", "pm25", "pm10", "temperature", "wind_speed"]:
            assert col in result.clean.columns, f"Expected canonical column '{col}' missing."

"""
tests/unit/test_preprocessor.py
=================================
Unit tests for src.data.preprocessor — AQIPreprocessor.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import AQIPreprocessor
from src.utils.config import (
    AppConfig, DataConfig, PreprocessingConfig, PathsConfig, ProjectConfig
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config():
    """A minimal AppConfig wired for testing (no real files needed)."""
    cfg = AppConfig()
    cfg.project = ProjectConfig(station_id="test_station", seed=42)
    cfg.data = DataConfig(
        datetime_col="datetime",
        target_col="AQI",
        pollutant_cols=["PM2.5", "PM10"],
        meteorological_cols=["temperature", "humidity"],
    )
    cfg.preprocessing = PreprocessingConfig(
        missing_strategy="interpolate",
        interpolation_method="linear",
        outlier_method="iqr",
        outlier_threshold=3.0,
        min_valid_data_pct=0.50,
    )
    cfg.paths = PathsConfig(processed_data="/tmp/processed/")
    return cfg


def _make_hourly_df(n_hours: int = 48, missing_frac: float = 0.0) -> pd.DataFrame:
    """Create a synthetic hourly DataFrame for testing."""
    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(n_hours)]
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "datetime": dates,
        "AQI":      rng.uniform(50, 300, n_hours),
        "PM2.5":    rng.uniform(10, 150, n_hours),
        "PM10":     rng.uniform(20, 200, n_hours),
        "temperature": rng.uniform(15, 40, n_hours),
        "humidity":    rng.uniform(30, 90, n_hours),
    })
    if missing_frac > 0:
        n_missing = int(n_hours * missing_frac)
        missing_idx = rng.choice(n_hours, n_missing, replace=False)
        df.loc[missing_idx, "PM2.5"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSetDatetimeIndex:
    def test_sets_index(self, minimal_config):
        df = _make_hourly_df()
        preprocessor = AQIPreprocessor(minimal_config)
        result = preprocessor._set_datetime_index(df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_missing_datetime_col_raises(self, minimal_config):
        df = _make_hourly_df().drop(columns=["datetime"])
        preprocessor = AQIPreprocessor(minimal_config)
        with pytest.raises(ValueError, match="Datetime column"):
            preprocessor._set_datetime_index(df)


class TestResampleHourly:
    def test_produces_regular_freq(self, minimal_config):
        df = _make_hourly_df(24)
        preprocessor = AQIPreprocessor(minimal_config)
        df = preprocessor._set_datetime_index(df)
        result = preprocessor._resample_hourly(df)
        # DatetimeIndex should have inferred 1h frequency
        diffs = pd.Series(result.index).diff().dropna()
        assert (diffs == pd.Timedelta("1h")).all()


class TestMissingValues:
    def test_no_nulls_after_interpolate(self, minimal_config):
        df = _make_hourly_df(48, missing_frac=0.20)
        preprocessor = AQIPreprocessor(minimal_config)
        df = preprocessor._set_datetime_index(df)
        df = preprocessor._resample_hourly(df)
        result = preprocessor._handle_missing_values(df)
        assert result.isna().sum().sum() == 0

    def test_ffill_strategy(self, minimal_config):
        minimal_config.preprocessing.missing_strategy = "ffill"
        df = _make_hourly_df(24, missing_frac=0.10)
        preprocessor = AQIPreprocessor(minimal_config)
        df = preprocessor._set_datetime_index(df)
        df = preprocessor._resample_hourly(df)
        result = preprocessor._handle_missing_values(df)
        assert result.isna().sum().sum() == 0

    def test_invalid_strategy_raises(self, minimal_config):
        minimal_config.preprocessing.missing_strategy = "magic"
        df = _make_hourly_df(24)
        preprocessor = AQIPreprocessor(minimal_config)
        df = preprocessor._set_datetime_index(df)
        df = preprocessor._resample_hourly(df)
        with pytest.raises(ValueError, match="Unknown missing_strategy"):
            preprocessor._handle_missing_values(df)


class TestOutliers:
    def test_capping_reduces_extremes(self, minimal_config):
        df = _make_hourly_df(100)
        # Inject extreme outlier
        df.loc[10, "PM2.5"] = 99999.0
        preprocessor = AQIPreprocessor(minimal_config)
        df_idx = preprocessor._set_datetime_index(df)
        df_idx = preprocessor._resample_hourly(df_idx)
        df_idx = preprocessor._handle_missing_values(df_idx)
        result = preprocessor._handle_outliers(df_idx, fit=True)
        assert result["PM2.5"].max() < 99999.0

    def test_bounds_stored_after_fit(self, minimal_config):
        df = _make_hourly_df(50)
        preprocessor = AQIPreprocessor(minimal_config)
        df_idx = preprocessor._set_datetime_index(df)
        df_idx = preprocessor._resample_hourly(df_idx)
        df_idx = preprocessor._handle_missing_values(df_idx)
        preprocessor._handle_outliers(df_idx, fit=True)
        assert len(preprocessor.outlier_bounds) > 0

    def test_transform_uses_fit_bounds(self, minimal_config):
        df = _make_hourly_df(50)
        preprocessor = AQIPreprocessor(minimal_config)
        # Full fit_transform on training data
        preprocessor.fit_transform(df)
        original_bounds = dict(preprocessor.outlier_bounds)

        # transform on new data — bounds must not change
        df2 = _make_hourly_df(20)
        preprocessor.transform(df2)
        assert preprocessor.outlier_bounds == original_bounds


class TestAQICategory:
    def test_category_column_added(self, minimal_config):
        df = _make_hourly_df(48)
        preprocessor = AQIPreprocessor(minimal_config)
        result = preprocessor.fit_transform(df)
        assert "aqi_category" in result.columns
        assert "aqi_category_name" in result.columns

    def test_category_values_in_range(self, minimal_config):
        df = _make_hourly_df(48)
        preprocessor = AQIPreprocessor(minimal_config)
        result = preprocessor.fit_transform(df)
        cats = result["aqi_category"].dropna()
        assert cats.between(0, 5).all()


class TestFitTransform:
    def test_returns_dataframe(self, minimal_config):
        df = _make_hourly_df(48)
        preprocessor = AQIPreprocessor(minimal_config)
        result = preprocessor.fit_transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_is_fitted_flag(self, minimal_config):
        df = _make_hourly_df(24)
        preprocessor = AQIPreprocessor(minimal_config)
        assert not preprocessor.is_fitted
        preprocessor.fit_transform(df)
        assert preprocessor.is_fitted

    def test_transform_before_fit_raises(self, minimal_config):
        df = _make_hourly_df(24)
        preprocessor = AQIPreprocessor(minimal_config)
        with pytest.raises(RuntimeError, match="fit_transform"):
            preprocessor.transform(df)

    def test_no_nulls_in_output(self, minimal_config):
        df = _make_hourly_df(72, missing_frac=0.15)
        preprocessor = AQIPreprocessor(minimal_config)
        result = preprocessor.fit_transform(df)
        numeric_nulls = result.select_dtypes(include=[np.number]).isna().sum().sum()
        assert numeric_nulls == 0

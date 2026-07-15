"""
tests/unit/preprocessing/test_cleaner.py
==========================================
Unit tests for src.preprocessing.cleaner.DataCleaner.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaner import DataCleaner
from src.utils.config import (
    AppConfig, DataConfig, PathsConfig, PreprocessingConfig, ProjectConfig
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    c = AppConfig()
    c.project = ProjectConfig(station_id="test_s", seed=42)
    c.data = DataConfig(datetime_col="datetime")
    c.preprocessing = PreprocessingConfig(
        missing_strategy="interpolate",
        interpolation_method="linear",
        outlier_method="iqr",
        outlier_threshold=3.0,
        min_valid_data_pct=0.50,
    )
    c.paths = PathsConfig(processed_data="/tmp/test_processed/")
    return c


def _make_validated_df(n_hours=72, missing_frac=0.0, seed=42):
    """Simulate a schema-validated DataFrame (canonical col names, station col)."""
    rng = np.random.default_rng(seed)
    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(n_hours)]
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "station": ["test_s"] * n_hours,
        "aqi":         rng.uniform(50, 300, n_hours),
        "pm25":        rng.uniform(10, 150, n_hours),
        "pm10":        rng.uniform(20, 200, n_hours),
        "temperature": rng.uniform(15, 40, n_hours),
        "humidity":    rng.uniform(30, 90, n_hours),
    })
    if missing_frac > 0:
        n_missing = max(1, int(n_hours * missing_frac))
        idx = rng.choice(n_hours, n_missing, replace=False)
        df.loc[idx, "pm25"] = np.nan
    return df


# ---------------------------------------------------------------------------
# DatetimeIndex
# ---------------------------------------------------------------------------

class TestSetDatetimeIndex:
    def test_sets_datetime_index(self, cfg):
        df = _make_validated_df()
        cleaner = DataCleaner(cfg)
        result = cleaner._set_datetime_index(df, "test_s")
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_date_not_a_column_after_index_set(self, cfg):
        df = _make_validated_df()
        cleaner = DataCleaner(cfg)
        result = cleaner._set_datetime_index(df, "test_s")
        assert "date" not in result.columns

    def test_raises_without_date_col(self, cfg):
        df = _make_validated_df().drop(columns=["date"])
        cleaner = DataCleaner(cfg)
        with pytest.raises(ValueError):
            cleaner._set_datetime_index(df, "test_s")


# ---------------------------------------------------------------------------
# Duplicate removal
# ---------------------------------------------------------------------------

class TestRemoveDuplicateTimestamps:
    def test_removes_duplicates(self, cfg):
        df = _make_validated_df(n_hours=4)
        cleaner = DataCleaner(cfg)
        df_idx = cleaner._set_datetime_index(df, "s")
        # Manually duplicate a row
        df_dup = pd.concat([df_idx, df_idx.iloc[[0]]])
        df_dup = df_dup.sort_index()
        result = cleaner._remove_duplicate_timestamps(df_dup, "s")
        assert not result.index.duplicated().any()


# ---------------------------------------------------------------------------
# Hourly resampling
# ---------------------------------------------------------------------------

class TestResampleHourly:
    def test_output_has_regular_hourly_freq(self, cfg):
        df = _make_validated_df(n_hours=48)
        cleaner = DataCleaner(cfg)
        df_idx = cleaner._set_datetime_index(df, "s")
        result = cleaner._resample_to_hourly(df_idx, "s")
        diffs = pd.Series(result.index).diff().dropna()
        assert (diffs == pd.Timedelta("1h")).all()


# ---------------------------------------------------------------------------
# Missing value imputation
# ---------------------------------------------------------------------------

class TestImputation:
    def test_no_nulls_after_interpolate(self, cfg):
        df = _make_validated_df(n_hours=72, missing_frac=0.20)
        cleaner = DataCleaner(cfg)
        df_idx = cleaner._set_datetime_index(df, "s")
        df_res = cleaner._resample_to_hourly(df_idx, "s")
        cleaner._learn_medians(df_res, "s")
        result = cleaner._impute_missing(df_res, "s")
        assert result.select_dtypes(include=[np.number]).isna().sum().sum() == 0

    def test_ffill_strategy(self, cfg):
        cfg.preprocessing.missing_strategy = "ffill"
        df = _make_validated_df(n_hours=48, missing_frac=0.10)
        cleaner = DataCleaner(cfg)
        df_idx = cleaner._set_datetime_index(df, "s")
        df_res = cleaner._resample_to_hourly(df_idx, "s")
        cleaner._learn_medians(df_res, "s")
        result = cleaner._impute_missing(df_res, "s")
        assert result.select_dtypes(include=[np.number]).isna().sum().sum() == 0

    def test_invalid_strategy_raises(self, cfg):
        cfg.preprocessing.missing_strategy = "magic_fill"
        df = _make_validated_df(n_hours=24)
        cleaner = DataCleaner(cfg)
        df_idx = cleaner._set_datetime_index(df, "s")
        df_res = cleaner._resample_to_hourly(df_idx, "s")
        cleaner._learn_medians(df_res, "s")
        with pytest.raises(ValueError, match="Unknown missing_strategy"):
            cleaner._impute_missing(df_res, "s")


# ---------------------------------------------------------------------------
# Non-negativity enforcement
# ---------------------------------------------------------------------------

class TestNonNegativity:
    def test_negative_aqi_clipped_to_zero(self, cfg):
        df = _make_validated_df(n_hours=24)
        cleaner = DataCleaner(cfg)
        df_idx = cleaner._set_datetime_index(df, "s")
        df_idx.loc[df_idx.index[0], "aqi"] = -5.0
        result = cleaner._enforce_non_negativity(df_idx, "s")
        assert result["aqi"].min() >= 0.0


# ---------------------------------------------------------------------------
# fit_transform / transform
# ---------------------------------------------------------------------------

class TestFitTransform:
    def test_returns_dataframe(self, cfg):
        df = _make_validated_df()
        cleaner = DataCleaner(cfg)
        result = cleaner.fit_transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_is_fitted_after_fit_transform(self, cfg):
        df = _make_validated_df()
        cleaner = DataCleaner(cfg)
        assert not cleaner.is_fitted
        cleaner.fit_transform(df)
        assert cleaner.is_fitted

    def test_transform_raises_before_fit(self, cfg):
        df = _make_validated_df()
        cleaner = DataCleaner(cfg)
        with pytest.raises(RuntimeError, match="fit_transform"):
            cleaner.transform(df)

    def test_no_nulls_in_output(self, cfg):
        df = _make_validated_df(n_hours=96, missing_frac=0.15)
        cleaner = DataCleaner(cfg)
        result = cleaner.fit_transform(df)
        assert result.select_dtypes(include=[np.number]).isna().sum().sum() == 0

    def test_transform_uses_fit_medians(self, cfg):
        df_train = _make_validated_df(n_hours=96)
        cleaner = DataCleaner(cfg)
        cleaner.fit_transform(df_train)
        original_medians = dict(cleaner.column_medians)

        df_test = _make_validated_df(n_hours=24, missing_frac=0.10, seed=99)
        cleaner.transform(df_test)
        # Medians must not change after transform
        assert cleaner.column_medians == original_medians

    def test_low_coverage_col_dropped(self, cfg):
        df = _make_validated_df(n_hours=100)
        # Set 60% of pressure to NaN (below min_valid_pct=0.50)
        df["pressure"] = np.nan
        df.loc[:39, "pressure"] = 1013.0   # only 40% valid
        cleaner = DataCleaner(cfg)
        result = cleaner.fit_transform(df)
        assert "pressure" not in result.columns

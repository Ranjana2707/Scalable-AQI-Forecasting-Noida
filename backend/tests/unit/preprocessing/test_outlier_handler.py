"""
tests/unit/preprocessing/test_outlier_handler.py
==================================================
Unit tests for src.preprocessing.outlier_handler.OutlierHandler.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.outlier_handler import OutlierHandler
from src.utils.config import (
    AppConfig, PathsConfig, PreprocessingConfig, ProjectConfig, DataConfig
)


@pytest.fixture
def cfg():
    c = AppConfig()
    c.project = ProjectConfig(station_id="test_s", seed=42)
    c.data = DataConfig()
    c.preprocessing = PreprocessingConfig(
        outlier_method="iqr",
        outlier_threshold=3.0,
        missing_strategy="interpolate",
        interpolation_method="linear",
        min_valid_data_pct=0.5,
    )
    c.paths = PathsConfig()
    return c


def _make_clean_df(n=200, seed=42):
    """Simulate a cleaned DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "aqi":  rng.uniform(50, 250, n),
        "pm25": rng.uniform(10, 120, n),
        "pm10": rng.uniform(20, 200, n),
    }, index=idx)


class TestIQRBounds:
    def test_bounds_computed(self, cfg):
        df = _make_clean_df()
        handler = OutlierHandler(cfg)
        lo, hi = handler._compute_bounds(df["aqi"].dropna(), "aqi")
        assert lo < hi

    def test_physical_bounds_applied(self, cfg):
        df = _make_clean_df()
        handler = OutlierHandler(cfg)
        # AQI physical max is 500 — statistical hi cannot exceed that
        lo, hi = handler._compute_bounds(df["aqi"].dropna(), "aqi")
        assert hi <= 500.0
        assert lo >= 0.0


class TestZScoreBounds:
    def test_zscore_bounds_computed(self, cfg):
        cfg.preprocessing.outlier_method = "zscore"
        df = _make_clean_df()
        handler = OutlierHandler(cfg)
        lo, hi = handler._compute_bounds(df["aqi"].dropna(), "aqi")
        assert lo < hi


class TestModifiedZScore:
    def test_modified_zscore_bounds_computed(self, cfg):
        cfg.preprocessing.outlier_method = "modified_zscore"
        df = _make_clean_df()
        handler = OutlierHandler(cfg)
        lo, hi = handler._compute_bounds(df["aqi"].dropna(), "aqi")
        assert lo < hi


class TestApplyTreatment:
    def test_capping_reduces_extreme_value(self, cfg):
        df = _make_clean_df(50)
        df.loc[df.index[0], "aqi"] = 99999.0
        handler = OutlierHandler(cfg)
        handler.fit_transform(df)
        assert df["aqi"].max() < 99999.0  # original df not changed (fit_transform copies)

    def test_fit_transform_caps_outliers(self, cfg):
        df = _make_clean_df(100)
        df.loc[df.index[5], "pm25"] = 50000.0
        handler = OutlierHandler(cfg)
        result = handler.fit_transform(df)
        assert result["pm25"].max() < 50000.0

    def test_bounds_stored_after_fit(self, cfg):
        df = _make_clean_df()
        handler = OutlierHandler(cfg)
        handler.fit_transform(df)
        assert len(handler.bounds) > 0

    def test_is_fitted_after_fit_transform(self, cfg):
        df = _make_clean_df()
        handler = OutlierHandler(cfg)
        assert not handler.is_fitted
        handler.fit_transform(df)
        assert handler.is_fitted

    def test_transform_raises_before_fit(self, cfg):
        df = _make_clean_df()
        handler = OutlierHandler(cfg)
        with pytest.raises(RuntimeError, match="fit_transform"):
            handler.transform(df)

    def test_transform_uses_fit_bounds(self, cfg):
        df_train = _make_clean_df(200)
        handler = OutlierHandler(cfg)
        handler.fit_transform(df_train)
        original_bounds = dict(handler.bounds)

        df_test = _make_clean_df(50, seed=99)
        handler.transform(df_test)
        assert handler.bounds == original_bounds


class TestOutlierSummary:
    def test_summary_returns_dataframe(self, cfg):
        df = _make_clean_df()
        df.loc[df.index[0], "aqi"] = 99999.0
        handler = OutlierHandler(cfg)
        handler.fit_transform(df)
        summary = handler.get_outlier_summary()
        assert isinstance(summary, pd.DataFrame)
        assert "column" in summary.columns
        assert "outliers_treated" in summary.columns

    def test_invalid_method_raises(self, cfg):
        cfg.preprocessing.outlier_method = "nonexistent_method"
        df = _make_clean_df(50)
        handler = OutlierHandler(cfg)
        with pytest.raises(ValueError, match="Unknown outlier_method"):
            handler.fit_transform(df)

"""
tests/unit/preprocessing/test_pipeline.py
==========================================
Unit tests for src.preprocessing.pipeline — split logic and pipeline result.

Note: Full end-to-end pipeline tests (with real file I/O) live in
tests/integration/.  These unit tests focus on the pure-logic components:
the time-series split function and the PreprocessingResult dataclass.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipeline import PreprocessingResult, _time_series_split
from src.utils.config import (
    AppConfig, DataConfig, PathsConfig, PreprocessingConfig,
    ProjectConfig, SplitConfig
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    c = AppConfig()
    c.project = ProjectConfig(station_id="test_s", seed=42)
    c.data = DataConfig()
    c.preprocessing = PreprocessingConfig()
    c.split = SplitConfig(
        strategy="time_series",
        test_size=0.15,
        val_size=0.10,
        gap_hours=24,
    )
    c.paths = PathsConfig(processed_data="/tmp/test_splits/")
    return c


def _make_hourly_df(n=500):
    """Simulate a fully cleaned DataFrame with a DatetimeIndex."""
    idx = pd.date_range("2022-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "station": ["test_s"] * n,
        "aqi":  rng.uniform(50, 300, n),
        "pm25": rng.uniform(10, 150, n),
    }, index=idx)


# ---------------------------------------------------------------------------
# Time-series split
# ---------------------------------------------------------------------------

class TestTimeSeriesSplit:
    def test_returns_three_dataframes(self, cfg):
        df = _make_hourly_df(500)
        train, val, test = _time_series_split(df, cfg, "test_s")
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val,   pd.DataFrame)
        assert isinstance(test,  pd.DataFrame)

    def test_sizes_sum_less_than_total(self, cfg):
        df = _make_hourly_df(500)
        train, val, test = _time_series_split(df, cfg, "test_s")
        # train + val + test + 2 gaps ≤ n
        total_used = len(train) + len(val) + len(test)
        assert total_used <= len(df)

    def test_chronological_order_preserved(self, cfg):
        df = _make_hourly_df(500)
        train, val, test = _time_series_split(df, cfg, "test_s")
        assert train.index.max() < val.index.min()
        assert val.index.max()   < test.index.min()

    def test_gap_between_train_and_val(self, cfg):
        df = _make_hourly_df(500)
        train, val, test = _time_series_split(df, cfg, "test_s")
        gap = (val.index.min() - train.index.max()).total_seconds() / 3600
        assert gap >= cfg.split.gap_hours

    def test_gap_between_val_and_test(self, cfg):
        df = _make_hourly_df(500)
        train, val, test = _time_series_split(df, cfg, "test_s")
        gap = (test.index.min() - val.index.max()).total_seconds() / 3600
        assert gap >= cfg.split.gap_hours

    def test_test_size_approx_correct(self, cfg):
        df = _make_hourly_df(1000)
        _, _, test = _time_series_split(df, cfg, "test_s")
        expected = int(1000 * cfg.split.test_size)
        assert abs(len(test) - expected) <= 2   # within 2 rows of target

    def test_val_size_approx_correct(self, cfg):
        df = _make_hourly_df(1000)
        _, val, _ = _time_series_split(df, cfg, "test_s")
        expected = int(1000 * cfg.split.val_size)
        assert abs(len(val) - expected) <= 2

    def test_no_data_overlap_between_splits(self, cfg):
        df = _make_hourly_df(500)
        train, val, test = _time_series_split(df, cfg, "test_s")
        train_idx = set(train.index)
        val_idx   = set(val.index)
        test_idx  = set(test.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_too_small_dataset_raises(self, cfg):
        df = _make_hourly_df(10)   # way too small for the configured splits
        with pytest.raises(ValueError, match="too small"):
            _time_series_split(df, cfg, "test_s")

    def test_train_is_largest_split(self, cfg):
        df = _make_hourly_df(500)
        train, val, test = _time_series_split(df, cfg, "test_s")
        assert len(train) > len(val)
        assert len(train) > len(test)


# ---------------------------------------------------------------------------
# PreprocessingResult dataclass
# ---------------------------------------------------------------------------

class TestPreprocessingResult:
    def _make_result(self):
        df = _make_hourly_df(200)
        return PreprocessingResult(
            raw=df.copy(),
            clean=df.copy(),
            train=df.iloc[:120].copy(),
            val=df.iloc[140:160].copy(),
            test=df.iloc[180:].copy(),
            station_id="test_s",
            elapsed_seconds=1.23,
        )

    def test_attributes_accessible(self):
        r = self._make_result()
        assert r.station_id == "test_s"
        assert len(r.train) == 120
        assert len(r.val)   == 20
        assert r.elapsed_seconds == pytest.approx(1.23)

    def test_dataframes_are_independent_copies(self):
        r = self._make_result()
        # Modifying train must not affect clean
        original_val = r.clean["aqi"].iloc[0]
        r.train.loc[r.train.index[0], "aqi"] = -9999.0
        assert r.clean["aqi"].iloc[0] == pytest.approx(original_val)

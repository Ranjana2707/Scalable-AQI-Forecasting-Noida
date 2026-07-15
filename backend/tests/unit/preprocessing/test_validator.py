"""
tests/unit/preprocessing/test_validator.py
============================================
Unit tests for src.preprocessing.validator.SchemaValidator.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.validator import (
    CANONICAL_SCHEMA,
    COLUMN_ALIASES,
    SchemaValidator,
    ValidationReport,
)
from src.utils.config import AppConfig, DataConfig, ProjectConfig, PathsConfig


@pytest.fixture
def cfg():
    c = AppConfig()
    c.project = ProjectConfig(station_id="test_s", seed=42)
    c.data = DataConfig(datetime_col="datetime")
    c.paths = PathsConfig()
    return c


def _make_raw_df(n=24):
    """Create a minimal valid raw DataFrame."""
    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(n)]
    return pd.DataFrame({
        "datetime": dates,
        "station": ["test_s"] * n,
        "aqi": np.random.uniform(50, 300, n),
        "pm25": np.random.uniform(10, 150, n),
    })


# ---------------------------------------------------------------------------
# Alias renaming
# ---------------------------------------------------------------------------

class TestRenameAliases:
    def test_pm25_alias_renamed(self, cfg):
        df = pd.DataFrame({"datetime": [datetime(2023,1,1)], "station": ["s"],
                           "aqi": [100.0], "PM2.5": [50.0]})
        v = SchemaValidator(cfg)
        result = v._rename_aliases(df)
        assert "pm25" in result.columns
        assert "PM2.5" not in result.columns

    def test_aqi_alias_renamed(self, cfg):
        df = pd.DataFrame({"datetime": [datetime(2023,1,1)], "station": ["s"],
                           "AQI": [100.0]})
        v = SchemaValidator(cfg)
        result = v._rename_aliases(df)
        assert "aqi" in result.columns

    def test_unknown_col_unchanged(self, cfg):
        df = pd.DataFrame({"datetime": [datetime(2023,1,1)], "aqi": [100.0],
                           "my_custom_col": [1.0]})
        v = SchemaValidator(cfg)
        result = v._rename_aliases(df)
        assert "my_custom_col" in result.columns


# ---------------------------------------------------------------------------
# Required column checks
# ---------------------------------------------------------------------------

class TestRequiredColumns:
    def test_missing_required_raises(self, cfg):
        df = pd.DataFrame({"station": ["s"], "aqi": [100.0]})  # no 'date'/'datetime'
        v = SchemaValidator(cfg)
        with pytest.raises(ValueError, match="Schema validation failed"):
            v.validate(df)

    def test_missing_optional_does_not_raise(self, cfg):
        df = _make_raw_df()
        v = SchemaValidator(cfg)
        result = v.validate(df)  # no exception
        assert result is not None

    def test_missing_optional_added_as_nan(self, cfg):
        df = _make_raw_df()
        v = SchemaValidator(cfg)
        result = v.validate(df)
        # e.g. "temperature" is optional and was not in df — should be NaN col
        assert "temperature" in result.columns
        assert result["temperature"].isna().all()


# ---------------------------------------------------------------------------
# Dtype coercion
# ---------------------------------------------------------------------------

class TestDtypeCoercion:
    def test_string_numeric_coerced_to_float(self, cfg):
        df = _make_raw_df()
        df["aqi"] = df["aqi"].astype(str)  # simulate string column
        v = SchemaValidator(cfg)
        result = v.validate(df)
        assert pd.api.types.is_float_dtype(result["aqi"])

    def test_non_numeric_string_becomes_nan(self, cfg):
        df = _make_raw_df()
        df.loc[0, "aqi"] = "N/A"
        df["aqi"] = df["aqi"].astype(str)
        v = SchemaValidator(cfg)
        result = v.validate(df)
        assert result["aqi"].isna().sum() >= 1

    def test_datetime_col_coerced(self, cfg):
        df = _make_raw_df()
        df["datetime"] = df["datetime"].astype(str)  # string datetime
        v = SchemaValidator(cfg)
        result = v.validate(df)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])


# ---------------------------------------------------------------------------
# Range validation
# ---------------------------------------------------------------------------

class TestRangeValidation:
    def test_out_of_range_logged_not_removed(self, cfg):
        df = _make_raw_df()
        df.loc[0, "aqi"] = 9999.0   # way above 500 max
        v = SchemaValidator(cfg)
        result = v.validate(df)
        # Value should still be there — range checks warn, not remove
        assert result["aqi"].max() == pytest.approx(9999.0)

    def test_range_violation_recorded_in_report(self, cfg):
        df = _make_raw_df()
        df.loc[0, "aqi"] = 9999.0
        v = SchemaValidator(cfg)
        v.validate(df)
        report = v.get_validation_report()
        assert "aqi" in report.range_violations


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

class TestValidationReport:
    def test_report_returned_after_validate(self, cfg):
        df = _make_raw_df()
        v = SchemaValidator(cfg)
        v.validate(df)
        report = v.get_validation_report()
        assert isinstance(report, ValidationReport)

    def test_report_raises_before_validate(self, cfg):
        v = SchemaValidator(cfg)
        with pytest.raises(RuntimeError):
            v.get_validation_report()

    def test_report_to_dict(self, cfg):
        df = _make_raw_df()
        v = SchemaValidator(cfg)
        v.validate(df)
        d = v.get_validation_report().to_dict()
        assert isinstance(d, dict)
        assert "station_id" in d

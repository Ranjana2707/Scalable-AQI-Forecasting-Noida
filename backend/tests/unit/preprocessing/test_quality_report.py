"""
tests/unit/preprocessing/test_quality_report.py
=================================================
Unit tests for src.preprocessing.quality_report.QualityReporter.
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.quality_report import QualityReport, QualityReporter
from src.utils.config import (
    AppConfig, DataConfig, PathsConfig, PreprocessingConfig, ProjectConfig
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    c = AppConfig()
    c.project = ProjectConfig(station_id="test_s", seed=42)
    c.data = DataConfig(
        datetime_col="datetime",
        target_col="aqi",
        aqi_breakpoints={
            "good":        [0,   50],
            "satisfactory":[51, 100],
            "moderate":    [101,200],
            "poor":        [201,300],
            "very_poor":   [301,400],
            "severe":      [401,500],
        },
    )
    c.preprocessing = PreprocessingConfig(min_valid_data_pct=0.5)
    c.paths = PathsConfig(reports=str(tmp_path / "reports") + "/")
    return c


def _make_raw_df(n=48):
    rng = np.random.default_rng(0)
    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(n)]
    return pd.DataFrame({
        "datetime": dates,
        "station": ["test_s"] * n,
        "aqi":  rng.uniform(50, 300, n),
        "pm25": rng.uniform(10, 150, n),
    })


def _make_clean_df(n=48):
    rng = np.random.default_rng(1)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "station": ["test_s"] * n,
        "aqi":  rng.uniform(50, 300, n),
        "pm25": rng.uniform(10, 150, n),
    }, index=idx)


# ---------------------------------------------------------------------------
# QualityReport container
# ---------------------------------------------------------------------------

class TestQualityReport:
    def test_to_dict_returns_dict(self):
        rpt = QualityReport("test_s")
        d = rpt.to_dict()
        assert isinstance(d, dict)
        assert "station_id" in d

    def test_print_summary_does_not_raise(self, cfg, capsys):
        reporter = QualityReporter(cfg)
        df_raw   = _make_raw_df()
        df_clean = _make_clean_df()
        rpt = reporter.generate(df_raw, df_clean, station_id="test_s")
        rpt.print_summary()   # should not raise
        captured = capsys.readouterr()
        assert "test_s" in captured.out


# ---------------------------------------------------------------------------
# QualityReporter.generate
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_returns_quality_report(self, cfg):
        reporter = QualityReporter(cfg)
        rpt = reporter.generate(_make_raw_df(), _make_clean_df())
        assert isinstance(rpt, QualityReport)

    def test_row_counts_populated(self, cfg):
        df_raw   = _make_raw_df(48)
        df_clean = _make_clean_df(48)
        reporter = QualityReporter(cfg)
        rpt = reporter.generate(df_raw, df_clean)
        assert rpt.total_rows_raw   == 48
        assert rpt.total_rows_clean == 48

    def test_date_range_populated(self, cfg):
        reporter = QualityReporter(cfg)
        rpt = reporter.generate(_make_raw_df(), _make_clean_df())
        assert rpt.date_range_start is not None
        assert rpt.date_range_end   is not None

    def test_temporal_coverage_100pct_regular_grid(self, cfg):
        reporter = QualityReporter(cfg)
        rpt = reporter.generate(_make_raw_df(), _make_clean_df(48))
        # Regular hourly grid with no gaps → 100 %
        assert rpt.temporal_coverage_pct == pytest.approx(100.0, abs=1.0)

    def test_missing_analysis_populated(self, cfg):
        df_clean = _make_clean_df()
        df_clean.loc[df_clean.index[:5], "pm25"] = np.nan
        reporter = QualityReporter(cfg)
        rpt = reporter.generate(_make_raw_df(), df_clean)
        cols = [r["column"] for r in rpt.missing_analysis]
        assert "pm25" in cols

    def test_aqi_dist_populated(self, cfg):
        reporter = QualityReporter(cfg)
        rpt = reporter.generate(_make_raw_df(), _make_clean_df())
        assert len(rpt.aqi_category_dist) > 0
        assert "Good" in rpt.aqi_category_dist

    def test_overall_score_between_0_and_100(self, cfg):
        reporter = QualityReporter(cfg)
        rpt = reporter.generate(_make_raw_df(), _make_clean_df())
        assert 0.0 <= rpt.overall_score <= 100.0

    def test_column_scores_all_in_range(self, cfg):
        reporter = QualityReporter(cfg)
        rpt = reporter.generate(_make_raw_df(), _make_clean_df())
        for col, score in rpt.column_scores.items():
            assert 0.0 <= score <= 100.0, f"{col} score {score} out of range"

    def test_get_report_raises_before_generate(self, cfg):
        reporter = QualityReporter(cfg)
        with pytest.raises(RuntimeError):
            reporter.get_report()


# ---------------------------------------------------------------------------
# QualityReporter.save
# ---------------------------------------------------------------------------

class TestSave:
    def test_json_file_created(self, cfg, tmp_path):
        reporter = QualityReporter(cfg)
        rpt = reporter.generate(_make_raw_df(), _make_clean_df())
        paths = reporter.save(rpt, station_id="test_s")
        assert paths["json"].exists()

    def test_html_file_created(self, cfg, tmp_path):
        reporter = QualityReporter(cfg)
        rpt = reporter.generate(_make_raw_df(), _make_clean_df())
        paths = reporter.save(rpt, station_id="test_s")
        assert paths["html"].exists()

    def test_html_contains_station_id(self, cfg, tmp_path):
        reporter = QualityReporter(cfg)
        rpt = reporter.generate(_make_raw_df(), _make_clean_df())
        paths = reporter.save(rpt, station_id="test_s")
        html_content = paths["html"].read_text(encoding="utf-8")
        assert "test_s" in html_content

    def test_save_raises_before_generate(self, cfg):
        reporter = QualityReporter(cfg)
        with pytest.raises(RuntimeError):
            reporter.save()


# ---------------------------------------------------------------------------
# Missing-value analysis helper
# ---------------------------------------------------------------------------

class TestAnalyseMissing:
    def test_zero_missing_gives_high_score(self, cfg):
        df_clean = _make_clean_df(48)
        reporter = QualityReporter(cfg)
        results = reporter._analyse_missing(df_clean, "test_s")
        for row in results:
            assert row["missing_count"] == 0
            assert row["quality_score"] == pytest.approx(100.0)

    def test_max_consecutive_run_detected(self, cfg):
        df_clean = _make_clean_df(48)
        # Create 5 consecutive NaNs
        df_clean.loc[df_clean.index[10:15], "aqi"] = np.nan
        reporter = QualityReporter(cfg)
        results = reporter._analyse_missing(df_clean, "test_s")
        aqi_row = next(r for r in results if r["column"] == "aqi")
        assert aqi_row["max_consecutive_missing"] == 5

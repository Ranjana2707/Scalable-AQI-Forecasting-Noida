"""
tests/unit/preprocessing/test_loader.py
=========================================
Unit tests for src.preprocessing.loader.DataLoader.
"""
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.loader import DataLoader, _DATETIME_FORMATS
from src.utils.config import AppConfig, DataConfig, PathsConfig, ProjectConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config(tmp_path):
    cfg = AppConfig()
    cfg.project = ProjectConfig(station_id="test_station", seed=42)
    cfg.data = DataConfig(datetime_col="datetime")
    cfg.paths = PathsConfig(
        raw_data=str(tmp_path / "raw") + "/",
        external_data=str(tmp_path / "external") + "/",
    )
    return cfg, tmp_path


def _write_csv(tmp_path, filename, content):
    p = tmp_path / "raw" / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# File discovery tests
# ---------------------------------------------------------------------------

class TestDiscoverFiles:
    def test_finds_station_prefixed_csv(self, minimal_config):
        cfg, tmp = minimal_config
        _write_csv(tmp, "test_station_2023.csv", """\
            datetime,AQI,PM2.5
            2023-01-01 00:00:00,150,75
            2023-01-01 01:00:00,160,80
        """)
        loader = DataLoader(cfg)
        files = loader._discover_files("test_station", None)
        assert len(files) == 1
        assert "test_station_2023.csv" in files[0]

    def test_fallback_to_any_csv_when_no_match(self, minimal_config):
        cfg, tmp = minimal_config
        _write_csv(tmp, "some_other_file.csv", """\
            datetime,AQI
            2023-01-01 00:00:00,100
        """)
        loader = DataLoader(cfg)
        files = loader._discover_files("nonexistent_station", None)
        assert len(files) == 1

    def test_raises_when_raw_dir_empty(self, minimal_config):
        cfg, tmp = minimal_config
        (tmp / "raw").mkdir(parents=True, exist_ok=True)
        loader = DataLoader(cfg)
        with pytest.raises(FileNotFoundError, match="No data files found"):
            loader._discover_files("test_station", None)


# ---------------------------------------------------------------------------
# CSV reading tests
# ---------------------------------------------------------------------------

class TestReadCSV:
    def test_reads_simple_csv(self, minimal_config):
        cfg, tmp = minimal_config
        p = _write_csv(tmp, "test_station.csv", """\
            datetime,AQI,PM2.5
            2023-01-01 00:00:00,150,75
            2023-01-01 01:00:00,160,80
        """)
        loader = DataLoader(cfg)
        df = loader._read_csv(str(p))
        assert len(df) == 2
        assert "AQI" in df.columns

    def test_strips_column_whitespace(self, minimal_config):
        cfg, tmp = minimal_config
        p = _write_csv(tmp, "test_station.csv", """\
            " datetime "," AQI ","PM2.5"
            2023-01-01 00:00:00,150,75
        """)
        loader = DataLoader(cfg)
        df = loader._read_csv(str(p))
        df = loader._normalise_columns(df)
        # Whitespace stripped from column names
        assert all(c == c.strip() for c in df.columns)


# ---------------------------------------------------------------------------
# Datetime parsing tests
# ---------------------------------------------------------------------------

class TestParseDatetime:
    @pytest.mark.parametrize("fmt,value", [
        ("%Y-%m-%d %H:%M:%S", "2023-06-15 14:30:00"),
        ("%d-%m-%Y %H:%M",    "15-06-2023 14:30"),
        ("%d/%m/%Y %H:%M",    "15/06/2023 14:30"),
    ])
    def test_parses_known_formats(self, minimal_config, fmt, value):
        cfg, tmp = minimal_config
        df = pd.DataFrame({"datetime": [value], "AQI": [100.0]})
        loader = DataLoader(cfg)
        result = loader._parse_datetime_column(df, "test_station")
        assert pd.api.types.is_datetime64_any_dtype(result["datetime"])

    def test_raises_on_unknown_format(self, minimal_config):
        cfg, _ = minimal_config
        df = pd.DataFrame({"datetime": ["Jun 15 2023 2:30PM"], "AQI": [100.0]})
        loader = DataLoader(cfg)
        with pytest.raises(ValueError, match="Cannot parse datetime"):
            loader._parse_datetime_column(df, "test_station")

    def test_raises_when_column_missing(self, minimal_config):
        cfg, _ = minimal_config
        df = pd.DataFrame({"AQI": [100.0]})
        loader = DataLoader(cfg)
        with pytest.raises(ValueError, match="not found"):
            loader._parse_datetime_column(df, "test_station")


# ---------------------------------------------------------------------------
# Station column tests
# ---------------------------------------------------------------------------

class TestAttachStationColumn:
    def test_adds_station_column_when_absent(self, minimal_config):
        cfg, _ = minimal_config
        df = pd.DataFrame({"datetime": ["2023-01-01"], "AQI": [100]})
        loader = DataLoader(cfg)
        result = loader._attach_station_column(df, "test_station")
        assert "station" in result.columns
        assert (result["station"] == "test_station").all()

    def test_preserves_existing_station_column(self, minimal_config):
        cfg, _ = minimal_config
        df = pd.DataFrame({
            "datetime": ["2023-01-01"],
            "AQI": [100],
            "station": ["original_name"],
        })
        loader = DataLoader(cfg)
        result = loader._attach_station_column(df, "new_name")
        assert result["station"].iloc[0] == "original_name"


# ---------------------------------------------------------------------------
# Sort and deduplicate tests
# ---------------------------------------------------------------------------

class TestSortAndDeduplicate:
    def test_removes_duplicate_timestamps(self, minimal_config):
        cfg, _ = minimal_config
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"]),
            "station": ["s", "s", "s"],
            "AQI": [100, 100, 110],
        })
        loader = DataLoader(cfg)
        result = loader._sort_and_deduplicate(df, "test_station")
        assert len(result) == 2

    def test_sorts_chronologically(self, minimal_config):
        cfg, _ = minimal_config
        df = pd.DataFrame({
            "datetime": pd.to_datetime(["2023-01-03", "2023-01-01", "2023-01-02"]),
            "station": ["s", "s", "s"],
            "AQI": [130, 100, 110],
        })
        loader = DataLoader(cfg)
        result = loader._sort_and_deduplicate(df, "s")
        assert result["datetime"].is_monotonic_increasing

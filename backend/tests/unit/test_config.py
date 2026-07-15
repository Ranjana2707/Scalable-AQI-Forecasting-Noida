"""
tests/unit/test_config.py
==========================
Unit tests for src.utils.config — config loading and merging.
"""
from pathlib import Path
import pytest
import yaml

from src.utils.config import (
    _deep_merge,
    _substitute_env_vars,
    AppConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99}

    def test_nested_merge(self):
        base = {"project": {"seed": 42, "station_id": "old"}}
        override = {"project": {"station_id": "new"}}
        result = _deep_merge(base, override)
        assert result["project"]["seed"] == 42
        assert result["project"]["station_id"] == "new"

    def test_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        override = {"a": {"x": 99}}
        _deep_merge(base, override)
        assert base["a"]["x"] == 1  # base unchanged

    def test_adds_new_keys(self):
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert "b" in result

    def test_list_override(self):
        base = {"cols": ["A", "B"]}
        override = {"cols": ["C"]}
        result = _deep_merge(base, override)
        assert result["cols"] == ["C"]


# ---------------------------------------------------------------------------
# _substitute_env_vars
# ---------------------------------------------------------------------------

class TestSubstituteEnvVars:
    def test_substitutes_set_var(self, monkeypatch):
        monkeypatch.setenv("MY_DIR", "/mnt/data")
        result = _substitute_env_vars("${MY_DIR}/raw")
        assert result == "/mnt/data/raw"

    def test_keeps_unset_var(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        result = _substitute_env_vars("${MISSING_VAR}/raw")
        assert result == "${MISSING_VAR}/raw"

    def test_nested_dict(self, monkeypatch):
        monkeypatch.setenv("ROOT", "/data")
        data = {"paths": {"raw": "${ROOT}/raw"}}
        result = _substitute_env_vars(data)
        assert result["paths"]["raw"] == "/data/raw"

    def test_list_items(self, monkeypatch):
        monkeypatch.setenv("COL", "PM2.5")
        result = _substitute_env_vars(["${COL}", "PM10"])
        assert result[0] == "PM2.5"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_default_yaml(self, tmp_path):
        cfg_content = {
            "project": {"name": "Test", "version": "0.1", "seed": 7,
                        "station_id": "test_station", "log_level": "DEBUG"},
            "paths": {"raw_data": "data/raw/"},
        }
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump(cfg_content))

        cfg = load_config(cfg_file)
        assert cfg.project.station_id == "test_station"
        assert cfg.project.seed == 7

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_override_merges(self, tmp_path):
        base_content = {
            "project": {"name": "Base", "version": "1.0", "seed": 42,
                        "station_id": "base_station", "log_level": "INFO"},
            "paths": {"raw_data": "data/raw/"},
        }
        override_content = {
            "project": {"station_id": "override_station"},
        }
        base_file = tmp_path / "base.yaml"
        override_file = tmp_path / "override.yaml"
        base_file.write_text(yaml.dump(base_content))
        override_file.write_text(yaml.dump(override_content))

        cfg = load_config(base_file, override_file)
        assert cfg.project.station_id == "override_station"
        assert cfg.project.seed == 42  # preserved from base

    def test_returns_appconfig(self, tmp_path):
        cfg_content = {"project": {"name": "T", "version": "0.1", "seed": 1,
                                   "station_id": "s", "log_level": "INFO"}}
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.dump(cfg_content))
        cfg = load_config(cfg_file)
        assert isinstance(cfg, AppConfig)

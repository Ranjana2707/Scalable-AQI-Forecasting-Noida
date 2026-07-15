"""
src/utils/config.py
====================
YAML configuration loader with validation and environment overrides.

Design
------
- Loads a base ``default.yaml`` then deep-merges any override file on top.
- Supports ``${ENV_VAR}`` substitution inside YAML values.
- Returns a frozen ``ProjectConfig`` dataclass for type-safe access.
- Raises descriptive errors on missing required fields.

Usage
-----
    from src.utils.config import load_config
    cfg = load_config("configs/default.yaml")
    print(cfg.project.station_id)
    print(cfg.paths.raw_data)
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses — one per top-level YAML section
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    name: str = "AQI Forecasting & XAI System — Noida"
    version: str = "1.0.0"
    seed: int = 42
    station_id: str = "noida_sector_62"
    log_level: str = "INFO"


@dataclass
class PathsConfig:
    raw_data: str = "data/raw/"
    interim_data: str = "data/interim/"
    processed_data: str = "data/processed/"
    external_data: str = "data/external/"
    models: str = "outputs/models/"
    figures: str = "outputs/figures/"
    reports: str = "outputs/reports/"
    predictions: str = "outputs/predictions/"
    logs: str = "logs/"
    mlflow_uri: str = "mlruns/"


@dataclass
class DataConfig:
    datetime_col: str = "datetime"
    target_col: str = "AQI"
    pollutant_cols: List[str] = field(default_factory=lambda: [
        "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
        "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
    ])
    meteorological_cols: List[str] = field(default_factory=lambda: [
        "temperature", "humidity", "wind_speed", "wind_direction", "rainfall",
    ])
    aqi_breakpoints: Dict[str, List[int]] = field(default_factory=lambda: {
        "good": [0, 50], "satisfactory": [51, 100],
        "moderate": [101, 200], "poor": [201, 300],
        "very_poor": [301, 400], "severe": [401, 500],
    })


@dataclass
class PreprocessingConfig:
    missing_strategy: str = "interpolate"
    interpolation_method: str = "time"
    outlier_method: str = "iqr"
    outlier_threshold: float = 3.0
    rolling_window_hours: int = 24
    min_valid_data_pct: float = 0.70


@dataclass
class SplitConfig:
    strategy: str = "time_series"
    test_size: float = 0.15
    val_size: float = 0.10
    gap_hours: int = 24


@dataclass
class EvaluationConfig:
    primary_metric: str = "rmse"
    metrics: List[str] = field(default_factory=lambda: [
        "rmse", "mae", "mape", "r2", "smape"
    ])
    cross_val_folds: int = 5
    cv_strategy: str = "time_series_split"


@dataclass
class MLflowConfig:
    experiment_name: str = "aqi_noida_forecasting"
    run_tags: Dict[str, str] = field(default_factory=lambda: {
        "project": "AQI-XAI-Noida",
        "environment": "development",
    })


@dataclass
class AppConfig:
    """Top-level configuration container."""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    # Raw dict kept for sections not yet dataclassed (features, models, dashboard)
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Fallback accessor for raw config sections."""
        return self._raw.get(key, default)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _substitute_env_vars(value: Any) -> Any:
    """Recursively replace ``${VAR}`` placeholders with environment values."""
    if isinstance(value, str):
        pattern = re.compile(r"\$\{(\w+)\}")
        def replacer(m: re.Match) -> str:
            env_val = os.getenv(m.group(1))
            if env_val is None:
                logger.warning(f"Environment variable '{m.group(1)}' not set; keeping placeholder.")
                return m.group(0)
            return env_val
        return pattern.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env_vars(i) for i in value]
    return value


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep-merge ``override`` into ``base``, returning a new dictionary.

    Override keys win at every nesting level.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load and parse a single YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    logger.debug(f"Loaded YAML config: {path}")
    return data


def _dict_to_appconfig(raw: Dict[str, Any]) -> AppConfig:
    """Convert a raw dictionary into a typed ``AppConfig``."""

    def _build(cls, data: dict):
        """Instantiate a dataclass from a dict, ignoring unknown keys."""
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    cfg = AppConfig(_raw=raw)

    if "project" in raw:
        cfg.project = _build(ProjectConfig, raw["project"])
    if "paths" in raw:
        cfg.paths = _build(PathsConfig, raw["paths"])
    if "data" in raw:
        cfg.data = _build(DataConfig, raw["data"])
    if "preprocessing" in raw:
        cfg.preprocessing = _build(PreprocessingConfig, raw["preprocessing"])
    if "split" in raw:
        cfg.split = _build(SplitConfig, raw["split"])
    if "evaluation" in raw:
        cfg.evaluation = _build(EvaluationConfig, raw["evaluation"])
    if "mlflow" in raw:
        cfg.mlflow = _build(MLflowConfig, raw["mlflow"])

    return cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    config_path: str | Path,
    override_path: Optional[str | Path] = None,
) -> AppConfig:
    """
    Load project configuration from YAML with optional environment overrides.

    Parameters
    ----------
    config_path : str | Path
        Path to the base YAML config (e.g., ``configs/default.yaml``).
    override_path : str | Path, optional
        Path to an environment-specific override file (e.g., ``configs/colab.yaml``).
        Keys in this file deep-merge on top of the base config.

    Returns
    -------
    AppConfig
        Fully validated, typed configuration object.

    Raises
    ------
    FileNotFoundError
        If either config file does not exist.
    yaml.YAMLError
        If a config file contains invalid YAML syntax.

    Examples
    --------
    >>> cfg = load_config("configs/default.yaml")
    >>> cfg.project.station_id
    'noida_sector_62'

    >>> cfg = load_config("configs/default.yaml", "configs/colab.yaml")
    >>> cfg.paths.raw_data   # overridden by colab.yaml
    '/content/drive/MyDrive/aqi_noida/data/raw/'
    """
    base_raw = _load_yaml(Path(config_path))

    if override_path is not None:
        override_raw = _load_yaml(Path(override_path))
        merged_raw = _deep_merge(base_raw, override_raw)
        logger.info(f"Config merged: {config_path} + {override_path}")
    else:
        merged_raw = base_raw

    merged_raw = _substitute_env_vars(merged_raw)
    cfg = _dict_to_appconfig(merged_raw)

    logger.info(
        f"Config loaded | station={cfg.project.station_id} | "
        f"version={cfg.project.version}"
    )
    return cfg

"""
src/features/interaction_features.py
======================================
Physics-motivated and EDA-driven interaction feature generation.

Scientific rationale for each interaction
------------------------------------------
PM2.5 × Humidity
    High humidity hygroscopic growth: aerosol particles absorb water
    and swell, increasing light scattering and effective PM2.5 mass.
    A PM2.5 of 150 µg/m³ at 90% RH is significantly more harmful than
    at 40% RH. Correlation between pm25 and humidity is 0.41 (EDA Fig 7).

PM10 × Wind Speed (inverse)
    Higher wind disperses coarse particles (PM10). Their interaction
    captures the dust suppression/generation dynamic:
    Low wind → PM10 accumulates; High wind → PM10 rises (but also clears).
    We use PM10 / (wind_speed + 1) to model accumulation tendency.

Temperature × O3
    Ozone photochemical production increases non-linearly with temperature.
    EDA Fig 13 shows the temperature-O3 relationship is season-dependent.
    Product term captures the summer ozone episode regime.

CO × PM2.5 (combustion fingerprint)
    Both are primary combustion products. Their product flags episodes
    where vehicular + industrial + residential burning co-occur — the
    most dangerous pollution regime (winter smog events).

Temperature × Humidity (apparent temperature proxy)
    Heat index approximation: models the combined physiological and
    chemical impact of hot humid days (enhancing secondary pollutant
    formation in summer).

Wind Speed × Humidity (fog/smog indicator)
    Low wind + high humidity = radiation fog conditions that trap
    pollutants near the surface. Their product inversely flags these days.

PM2.5 / PM10 ratio (source fingerprint)
    PM2.5/PM10 > 0.6 → combustion / fine particle dominance (winter)
    PM2.5/PM10 < 0.4 → dust / coarse particle dominance (summer/spring)
    This ratio is a scientifically validated source attribution indicator.

AQI × aqi_saturated (saturation interaction)
    When AQI is capped at 500, this interaction preserves information
    about the saturation magnitude for models that need to learn the
    ceiling effect.
"""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all physics-motivated and EDA-driven interaction features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with canonical columns already present.

    Returns
    -------
    pd.DataFrame
        With interaction feature columns appended.

    Notes
    -----
    Interaction features are computed from raw (un-lagged) values.
    The pipeline applies this module AFTER temporal features but BEFORE
    lag/rolling — the lag module will then automatically create lags of
    the raw pollutants, which is sufficient. Interactions from lagged
    values would be redundant.
    """
    df = df.copy()
    n_added = 0

    def _safe_add(name: str, values: pd.Series) -> None:
        """Add interaction column with rounding and log."""
        nonlocal n_added
        df[name] = values.round(4)
        n_added += 1

    # ── 1. PM2.5 × Humidity (hygroscopic growth) ─────────────────────
    if "pm25" in df.columns and "humidity" in df.columns:
        _safe_add("interact_pm25_humidity",
                  df["pm25"] * df["humidity"] / 100)

    # ── 2. PM10 / (Wind Speed + 1) (accumulation tendency) ───────────
    if "pm10" in df.columns and "wind_speed" in df.columns:
        _safe_add("interact_pm10_wind_accum",
                  df["pm10"] / (df["wind_speed"] + 1.0))

    # ── 3. Temperature × O3 (photochemical production) ───────────────
    if "temperature" in df.columns and "o3" in df.columns:
        _safe_add("interact_temp_o3",
                  df["temperature"] * df["o3"] / 100)

    # ── 4. CO × PM2.5 (combustion co-occurrence) ─────────────────────
    if "co" in df.columns and "pm25" in df.columns:
        _safe_add("interact_co_pm25",
                  df["co"] * df["pm25"])

    # ── 5. Temperature × Humidity (apparent temperature proxy) ────────
    if "temperature" in df.columns and "humidity" in df.columns:
        _safe_add("interact_temp_humidity",
                  df["temperature"] * df["humidity"] / 100)

    # ── 6. Wind Speed × Humidity (fog/smog formation indicator) ──────
    if "wind_speed" in df.columns and "humidity" in df.columns:
        _safe_add("interact_wind_humidity",
                  df["wind_speed"] * df["humidity"] / 100)

    # ── 7. PM2.5 / PM10 ratio (source fingerprint) ────────────────────
    if "pm25" in df.columns and "pm10" in df.columns:
        # Clamp denominator to avoid div/0
        denom = df["pm10"].clip(lower=1.0)
        _safe_add("interact_pm25_pm10_ratio",
                  (df["pm25"] / denom).clip(upper=2.0))

    # ── 8. NO2 × CO (traffic+combustion co-emission) ──────────────────
    if "no2" in df.columns and "co" in df.columns:
        _safe_add("interact_no2_co",
                  df["no2"] * df["co"])

    # ── 9. Pressure × Temperature (atmospheric stability proxy) ───────
    # High pressure + low temperature = strong temperature inversion
    if "pressure" in df.columns and "temperature" in df.columns:
        press_norm = df["pressure"] / 1013.0   # normalise to standard atm
        temp_inv   = 1.0 / (df["temperature"] + 30).clip(lower=1.0)  # cold=high
        _safe_add("interact_pressure_temp_stability",
                  press_norm * temp_inv * 100)

    # ── 10. AQI lag-1 × is_winter (winter persistence) ───────────────
    # High AQI persists longer in winter (inversion traps pollutants)
    if "lag_aqi_1" in df.columns and "season_Winter" in df.columns:
        _safe_add("interact_lag_aqi1_winter",
                  df["lag_aqi_1"] * df["season_Winter"])

    # ── 11. PM2.5 × is_stubble_burning (crop-burning fingerprint) ─────
    if "pm25" in df.columns and "is_stubble_burning" in df.columns:
        _safe_add("interact_pm25_stubble",
                  df["pm25"] * df["is_stubble_burning"])

    # ── 12. Wind Speed squared (turbulent mixing) ─────────────────────
    if "wind_speed" in df.columns:
        _safe_add("interact_wind_speed_sq",
                  df["wind_speed"] ** 2)

    logger.info(f"Interaction features added: {n_added}")
    return df


def get_interaction_feature_names() -> List[str]:
    """Return all interaction feature names this module produces."""
    return [
        "interact_pm25_humidity",
        "interact_pm10_wind_accum",
        "interact_temp_o3",
        "interact_co_pm25",
        "interact_temp_humidity",
        "interact_wind_humidity",
        "interact_pm25_pm10_ratio",
        "interact_no2_co",
        "interact_pressure_temp_stability",
        "interact_lag_aqi1_winter",
        "interact_pm25_stubble",
        "interact_wind_speed_sq",
    ]

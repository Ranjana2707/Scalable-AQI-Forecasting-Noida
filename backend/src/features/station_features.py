"""
src/features/station_features.py
==================================
Station identity, encoding, and station-specific statistical features.

EDA justification
-----------------
From EDA Fig 12 (inter-station scatter): r = 0.964 between stations,
with Sector-1 consistently ~8 AQI points higher than Sector-62.
This systematic offset means a simple integer encoding (0/1) is
insufficient for models that need to learn station-specific baselines.

Features produced
-----------------
station_encoded        : 0 = noida_sector_1, 1 = noida_sector_62
station_mean_aqi       : historical mean AQI per station (train-set only)
station_std_aqi        : historical std AQI per station (train-set only)
station_mean_pm25      : historical mean PM2.5 per station
station_aqi_offset     : station deviation from grand mean
station_season_mean_aqi: station × season interaction mean AQI
  → creates 5 features: one per season

Leakage prevention
------------------
Station statistics (mean, std) are computed on TRAINING data only and
applied to validation/test via a lookup table. The ``fit`` method
learns the statistics; ``transform`` applies them. This is the same
fit/transform pattern used throughout the preprocessing pipeline.

Usage
-----
    encoder = StationEncoder(config)
    encoder.fit(train_df)
    train_df = encoder.transform(train_df)
    val_df   = encoder.transform(val_df)
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Station encoding map (consistent with configs/default.yaml)
STATION_ENCODING: Dict[str, int] = {
    "noida_sector_1":  0,
    "noida_sector_62": 1,
}

SEASONS = ["Winter", "Spring", "Summer", "Monsoon", "Post_Monsoon"]


class StationEncoder:
    """
    Stateful station feature generator.

    Learns per-station statistics on training data and applies them
    to any split without leakage.

    Parameters
    ----------
    station_col : str
        Column containing station identifiers.
    encoding_map : dict, optional
        Maps station string → integer. Defaults to STATION_ENCODING.

    Attributes
    ----------
    station_stats_ : pd.DataFrame
        Per-station statistics learned during ``fit()``.
    season_stats_  : pd.DataFrame
        Per-station × season statistics learned during ``fit()``.
    grand_mean_aqi_: float
        Overall mean AQI across all training rows.
    is_fitted_     : bool
    """

    def __init__(
        self,
        station_col: str = "station",
        encoding_map: Optional[Dict[str, int]] = None,
    ) -> None:
        self.station_col  = station_col
        self.encoding_map = encoding_map or STATION_ENCODING
        self.station_stats_: Optional[pd.DataFrame] = None
        self.season_stats_:  Optional[pd.DataFrame] = None
        self.grand_mean_aqi_: float = 0.0
        self.is_fitted_: bool = False

    def fit(self, train_df: pd.DataFrame) -> "StationEncoder":
        """
        Learn station statistics from training data.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training split only.

        Returns
        -------
        self
        """
        logger.info(
            f"StationEncoder.fit | stations={train_df[self.station_col].unique().tolist()} | "
            f"rows={len(train_df)}"
        )

        # Per-station aggregate statistics
        stats = train_df.groupby(self.station_col, observed=True).agg(
            station_mean_aqi=("aqi", "mean"),
            station_std_aqi=("aqi", "std"),
            station_mean_pm25=("pm25", "mean"),
            station_mean_pm10=("pm10", "mean"),
            station_mean_temp=("temperature", "mean"),
        ).round(4)
        self.station_stats_ = stats
        self.grand_mean_aqi_ = float(train_df["aqi"].mean())

        # Per-station × season mean AQI
        if "season" in train_df.columns:
            self.season_stats_ = (
                train_df.groupby([self.station_col, "season"], observed=True)["aqi"]
                        .mean().round(4).unstack(fill_value=self.grand_mean_aqi_)
            )
        else:
            self.season_stats_ = None

        self.is_fitted_ = True
        logger.info(f"Station statistics learned:\n{self.station_stats_}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add station features to a DataFrame using learned statistics.

        Parameters
        ----------
        df : pd.DataFrame
            Any split (train/val/test) after ``fit()`` has been called.

        Returns
        -------
        pd.DataFrame
            With station feature columns appended.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() on training data before transform().")

        df = df.copy()
        sc = self.station_col

        # 1. Integer encoding
        df["station_encoded"] = (
            df[sc].map(self.encoding_map).fillna(-1).astype("int8")
        )

        # 2. Lookup-based statistics
        df = df.join(self.station_stats_, on=sc, how="left")

        # 3. Station AQI offset from grand mean
        df["station_aqi_offset"] = (
            df["station_mean_aqi"] - self.grand_mean_aqi_
        ).round(4)

        # 4. Station × season mean AQI
        if self.season_stats_ is not None and "season" in df.columns:
            for season in SEASONS:
                col_name = f"station_season_mean_{season.lower()}"
                if season in self.season_stats_.columns:
                    df[col_name] = df[sc].map(
                        self.season_stats_[season].to_dict()
                    ).fillna(self.grand_mean_aqi_)
                else:
                    df[col_name] = self.grand_mean_aqi_

        logger.info(f"Station features applied | shape={df.shape}")
        return df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit on and transform the training data."""
        return self.fit(train_df).transform(train_df)

    def get_feature_names(self) -> List[str]:
        """Return all station feature column names this encoder produces."""
        base = [
            "station_encoded",
            "station_mean_aqi",
            "station_std_aqi",
            "station_mean_pm25",
            "station_mean_pm10",
            "station_mean_temp",
            "station_aqi_offset",
        ]
        season_feats = [
            f"station_season_mean_{s.lower()}" for s in SEASONS
        ]
        return base + season_feats

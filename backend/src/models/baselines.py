"""
src/models/baselines.py
========================
Naive persistence baselines — the minimum benchmark every model must beat.

Persistence lag-1  : AQI_pred[t] = AQI[t-1]
Persistence lag-7  : AQI_pred[t] = AQI[t-7]
Seasonal mean      : AQI_pred[t] = monthly mean from training data

These are the standard benchmarks for daily air quality forecasting
(Kumar & Goyal, 2011; Ong et al., 2016).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional
from src.models.metrics import evaluate_all


class PersistenceBaseline:
    """
    Naive lag-k persistence forecaster.

    Predicts AQI[t] = AQI[t-k].  Uses the pre-computed lag features
    already present in the feature DataFrame (lag_aqi_1, lag_aqi_7).

    Parameters
    ----------
    lag : int
        Lag period (1 = yesterday, 7 = same day last week).
    """

    def __init__(self, lag: int = 1) -> None:
        self.lag = lag
        self.name = f"Persistence_lag{lag}"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        col = f"lag_aqi_{self.lag}"
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Run lag feature engineering first.")
        return df[col].values

    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = "aqi",
        station_col: str = "station",
    ) -> Dict:
        """Return per-station + combined evaluation dict."""
        preds = self.predict(df)
        y     = df[target_col].values
        results = {"model": self.name, "combined": evaluate_all(y, preds)}

        for sid, grp in df.groupby(station_col, observed=True):
            idx = grp.index
            results[sid] = evaluate_all(y[idx - df.index[0]], preds[idx - df.index[0]])

        return results


class SeasonalMeanBaseline:
    """
    Predicts AQI[t] = training-set mean AQI for the same month.
    Fit on training data; transform on any split.
    """

    def __init__(self) -> None:
        self.name = "Seasonal_Mean"
        self._monthly_means: Optional[Dict] = None

    def fit(self, train_df: pd.DataFrame, target_col: str = "aqi") -> "SeasonalMeanBaseline":
        self._monthly_means = (
            train_df.groupby("month")[target_col].mean().to_dict()
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._monthly_means is None:
            raise RuntimeError("Call fit() before predict().")
        return df["month"].map(self._monthly_means).values

    def evaluate(self, df, target_col="aqi", station_col="station") -> Dict:
        preds = self.predict(df)
        y = df[target_col].values
        results = {"model": self.name, "combined": evaluate_all(y, preds)}
        for sid, grp in df.groupby(station_col, observed=True):
            mask = df[station_col] == sid
            results[sid] = evaluate_all(y[mask], preds[mask])
        return results

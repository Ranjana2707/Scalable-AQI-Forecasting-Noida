"""
src/deep_learning/sequence_builder.py
========================================
Converts the Phase 4 tabular feature dataset into 3D sequence tensors
(samples, timesteps, features) for sequence models, with configurable
window size and forecast horizon, applied per-station to avoid
sequence leakage across station boundaries.

Design
------
- Window size (lookback): how many past days the model sees.
- Horizon: how many days ahead to predict (1 = next-day, >1 = multi-step).
- Sequences are built independently per station, then concatenated,
  so no sequence window ever spans two different stations.
- Scaling (zero mean, unit variance) is fit ONLY on the training split
  and applied to val/test — identical leakage protection to Phase 5.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ScalerStats:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse_transform_target(self, y: np.ndarray, target_idx: int) -> np.ndarray:
        return y * self.std[target_idx] + self.mean[target_idx]


class SequenceBuilder:
    """
    Builds sliding-window sequences per station for sequence models.

    Parameters
    ----------
    feature_cols : list of str
        Columns to include as model inputs (must include target_col).
    target_col : str
        Column to forecast.
    window_size : int
        Number of past timesteps (days) used as input context.
    horizon : int
        Number of days ahead to forecast (1 = next day).
    station_col : str
    date_col : str
    """

    def __init__(
        self,
        feature_cols: List[str],
        target_col: str = "aqi",
        window_size: int = 14,
        horizon: int = 1,
        station_col: str = "station",
        date_col: str = "date",
    ) -> None:
        self.feature_cols = feature_cols
        self.target_col   = target_col
        self.window_size  = window_size
        self.horizon      = horizon
        self.station_col  = station_col
        self.date_col     = date_col
        self.scaler: Optional[ScalerStats] = None
        self.target_idx = feature_cols.index(target_col) if target_col in feature_cols else None

    def fit_scaler(self, train_df: pd.DataFrame) -> "SequenceBuilder":
        """Fit z-score scaler on training data only."""
        X = train_df[self.feature_cols].values.astype(np.float64)
        mean = X.mean(axis=0)
        std  = X.std(axis=0)
        std[std == 0] = 1.0  # avoid div-by-zero for constant columns
        self.scaler = ScalerStats(mean=mean, std=std)
        return self

    def build(
        self,
        df: pd.DataFrame,
        return_meta: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
        """
        Build sequences from a DataFrame, scaled using the fitted scaler.

        Parameters
        ----------
        df : pd.DataFrame
            Sorted by (station, date). Must contain feature_cols.
        return_meta : bool
            If True, also return a metadata DataFrame with the
            (station, date) of each sequence's target.

        Returns
        -------
        X : np.ndarray, shape (n_samples, window_size, n_features)
        y : np.ndarray, shape (n_samples,)
        meta : pd.DataFrame or None — columns [station, date] for each y
        """
        if self.scaler is None:
            raise RuntimeError("Call fit_scaler() on training data before build().")

        all_X, all_y, all_meta = [], [], []

        for sid, grp in df.groupby(self.station_col, observed=True):
            grp = grp.sort_values(self.date_col).reset_index(drop=True)
            feat_mat = grp[self.feature_cols].values.astype(np.float64)
            feat_mat_scaled = self.scaler.transform(feat_mat)
            target_raw = grp[self.target_col].values.astype(np.float64)
            dates = grp[self.date_col].values

            n = len(grp)
            max_start = n - self.window_size - self.horizon + 1
            if max_start <= 0:
                continue

            for start in range(max_start):
                end = start + self.window_size
                target_idx = end + self.horizon - 1
                all_X.append(feat_mat_scaled[start:end])
                all_y.append(target_raw[target_idx])
                all_meta.append((sid, dates[target_idx]))

        X = np.array(all_X, dtype=np.float64)
        y = np.array(all_y, dtype=np.float64)
        meta = None
        if return_meta:
            meta = pd.DataFrame(all_meta, columns=[self.station_col, self.date_col])

        return X, y, meta

    def get_n_features(self) -> int:
        return len(self.feature_cols)

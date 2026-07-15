"""
src/models/linear_models.py
=============================
Regularised linear regression baselines (Ridge, Lasso, ElasticNet).

All models are wrapped in a sklearn Pipeline with StandardScaler so
the feature scale differences (AQI 0–500 vs sin_month −1 to 1) do
not bias the coefficient magnitudes.

Hyperparameters are set to well-known defaults then tuned lightly
via validation-set grid search over alpha values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.models.metrics import evaluate_all
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LinearAQIModel:
    """
    Wrapper for regularised linear models with StandardScaler.

    Parameters
    ----------
    model_type : str
        One of 'ridge', 'lasso', 'elasticnet'.
    alpha : float
        Regularisation strength.
    l1_ratio : float
        ElasticNet mixing parameter (ignored for Ridge/Lasso).
    """

    MODEL_MAP = {
        "ridge":      Ridge,
        "lasso":      Lasso,
        "elasticnet": ElasticNet,
    }

    def __init__(
        self,
        model_type: str = "ridge",
        alpha: float = 10.0,
        l1_ratio: float = 0.5,
        max_iter: int = 5000,
    ) -> None:
        if model_type not in self.MODEL_MAP:
            raise ValueError(f"Unknown model_type '{model_type}'. "
                             f"Choose from: {list(self.MODEL_MAP.keys())}")
        self.model_type = model_type
        self.alpha      = alpha
        self.l1_ratio   = l1_ratio
        self.name       = f"{model_type.capitalize()}(α={alpha})"
        kwargs = {"alpha": alpha, "max_iter": max_iter}
        if model_type == "elasticnet":
            kwargs["l1_ratio"] = l1_ratio
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  self.MODEL_MAP[model_type](**kwargs)),
        ])
        self.feature_cols: Optional[List[str]] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "aqi",
    ) -> "LinearAQIModel":
        self.feature_cols = feature_cols
        X = train_df[feature_cols].values
        y = train_df[target_col].values
        self.pipeline.fit(X, y)
        logger.info(f"{self.name} fitted | train_rows={len(train_df)}")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].values
        return self.pipeline.predict(X)

    def evaluate(
        self, df: pd.DataFrame,
        target_col: str = "aqi",
        station_col: str = "station",
    ) -> Dict:
        preds = self.predict(df)
        y = df[target_col].values
        results = {"model": self.name, "combined": evaluate_all(y, preds)}
        for sid, grp in df.groupby(station_col, observed=True):
            mask = df[station_col] == sid
            results[sid] = evaluate_all(y[mask.values], preds[mask.values])
        return results

    def get_top_coefficients(self, n: int = 20) -> pd.DataFrame:
        """Return the top-n features by absolute coefficient magnitude."""
        coefs = self.pipeline.named_steps["model"].coef_
        feat_df = pd.DataFrame({
            "feature": self.feature_cols,
            "coefficient": coefs,
            "abs_coef": np.abs(coefs),
        }).sort_values("abs_coef", ascending=False).head(n)
        return feat_df

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        logger.info(f"{self.name} saved → {path}")

    def load(self, path: str) -> "LinearAQIModel":
        self.pipeline = joblib.load(path)
        logger.info(f"{self.name} loaded ← {path}")
        return self


def tune_alpha(
    model_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    alphas: Optional[List[float]] = None,
    target_col: str = "aqi",
) -> Tuple[float, pd.DataFrame]:
    """
    Grid-search alpha on the validation set; return best alpha and results table.

    Parameters
    ----------
    model_type : str
    train_df, val_df : pd.DataFrame
    feature_cols : list of str
    alphas : list of float, optional
        Candidates. Defaults to [0.01, 0.1, 1, 10, 50, 100, 500].
    target_col : str

    Returns
    -------
    best_alpha : float
    results_df : pd.DataFrame — one row per alpha with val RMSE
    """
    alphas = alphas or [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0]
    rows = []
    for a in alphas:
        m = LinearAQIModel(model_type=model_type, alpha=a)
        m.fit(train_df, feature_cols, target_col)
        metrics = m.evaluate(val_df, target_col)
        rows.append({"alpha": a,
                     "val_rmse": round(metrics["combined"]["rmse"], 4),
                     "val_r2":   round(metrics["combined"]["r2"],   4)})
        logger.debug(f"  {model_type} α={a}: RMSE={rows[-1]['val_rmse']}")

    results_df = pd.DataFrame(rows)
    best_alpha = float(results_df.loc[results_df["val_rmse"].idxmin(), "alpha"])
    logger.info(f"{model_type} best alpha={best_alpha} (val RMSE={results_df['val_rmse'].min():.3f})")
    return best_alpha, results_df

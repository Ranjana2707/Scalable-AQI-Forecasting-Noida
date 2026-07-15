"""
src/models/ensemble_models.py
===============================
Tree-based ensemble models: RandomForest, ExtraTrees,
HistGradientBoosting (sklearn's XGBoost-equivalent).

All three are native sklearn and require no additional dependencies.
HistGradientBoosting handles missing values natively and supports
early stopping — making it the most XGBoost-like of the three.

Feature importance is extracted from all models for SHAP preparation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor,
)

from src.models.metrics import evaluate_all
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleAQIModel:
    """
    Unified wrapper for tree-based ensemble models.

    Parameters
    ----------
    model_type : str
        One of: 'random_forest', 'extra_trees', 'hist_gradient_boosting',
        'gradient_boosting'.
    **kwargs
        Passed to the underlying sklearn estimator.
    """

    MODEL_MAP = {
        "random_forest":          RandomForestRegressor,
        "extra_trees":            ExtraTreesRegressor,
        "hist_gradient_boosting": HistGradientBoostingRegressor,
        "gradient_boosting":      GradientBoostingRegressor,
    }

    DEFAULT_PARAMS = {
        "random_forest": {
            "n_estimators": 300, "max_depth": 15,
            "min_samples_leaf": 3, "max_features": "sqrt",
            "n_jobs": -1, "random_state": 42,
        },
        "extra_trees": {
            "n_estimators": 300, "max_depth": 15,
            "min_samples_leaf": 3, "max_features": "sqrt",
            "n_jobs": -1, "random_state": 42,
        },
        "hist_gradient_boosting": {
            "max_iter": 500, "learning_rate": 0.05,
            "max_depth": 6, "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "early_stopping": True, "validation_fraction": 0.1,
            "n_iter_no_change": 30, "random_state": 42,
        },
        "gradient_boosting": {
            "n_estimators": 400, "learning_rate": 0.05,
            "max_depth": 5, "min_samples_leaf": 10,
            "subsample": 0.8, "max_features": "sqrt",
            "random_state": 42,
        },
    }

    def __init__(self, model_type: str = "hist_gradient_boosting", **kwargs) -> None:
        if model_type not in self.MODEL_MAP:
            raise ValueError(f"model_type must be one of {list(self.MODEL_MAP.keys())}")
        self.model_type   = model_type
        params            = {**self.DEFAULT_PARAMS[model_type], **kwargs}
        self.model        = self.MODEL_MAP[model_type](**params)
        self.name         = model_type.replace("_", " ").title()
        self.feature_cols: Optional[List[str]] = None
        self.params       = params

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "aqi",
        val_df: Optional[pd.DataFrame] = None,
    ) -> "EnsembleAQIModel":
        self.feature_cols = feature_cols
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values

        logger.info(f"Training {self.name} | "
                    f"train_rows={len(X_train)} | features={len(feature_cols)}")
        self.model.fit(X_train, y_train)

        # In-sample performance
        train_preds = self.model.predict(X_train)
        train_rmse  = np.sqrt(np.mean((y_train - train_preds)**2))
        logger.info(f"  Train RMSE: {train_rmse:.3f}")

        if val_df is not None:
            val_metrics = self.evaluate(val_df, target_col)
            logger.info(f"  Val RMSE: {val_metrics['combined']['rmse']:.3f} | "
                        f"Val R²: {val_metrics['combined']['r2']:.4f}")

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[self.feature_cols].values)

    def evaluate(
        self, df: pd.DataFrame,
        target_col: str = "aqi",
        station_col: str = "station",
    ) -> Dict:
        preds = self.predict(df)
        y     = df[target_col].values
        results = {"model": self.name, "combined": evaluate_all(y, preds)}
        for sid, grp in df.groupby(station_col, observed=True):
            mask = (df[station_col] == sid).values
            results[sid] = evaluate_all(y[mask], preds[mask])
        return results

    def feature_importance_df(self, top_n: int = 30) -> pd.DataFrame:
        """Return a ranked DataFrame of feature importances."""
        if not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame()
        importances = self.model.feature_importances_
        fi_df = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
        fi_df["rank"] = fi_df.index + 1
        return fi_df

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model,
                     "feature_cols": self.feature_cols,
                     "name": self.name,
                     "params": self.params}, path)
        logger.info(f"{self.name} saved → {path}")

    @classmethod
    def load(cls, path: str) -> "EnsembleAQIModel":
        data = joblib.load(path)
        obj  = cls.__new__(cls)
        obj.model        = data["model"]
        obj.feature_cols = data["feature_cols"]
        obj.name         = data["name"]
        obj.params       = data["params"]
        obj.model_type   = data["name"]
        logger.info(f"Model loaded ← {path}")
        return obj

"""
src/utils/metrics.py
=====================
Pure-function evaluation metrics for AQI regression forecasting.

All functions accept NumPy arrays or pandas Series and return a float.
No model objects, no side effects — fully testable in isolation.

Metrics implemented
-------------------
- RMSE  : Root Mean Squared Error
- MAE   : Mean Absolute Error
- MAPE  : Mean Absolute Percentage Error
- SMAPE : Symmetric Mean Absolute Percentage Error
- R2    : Coefficient of Determination
- evaluate_all : Returns a dict of all metrics in one call
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Union

ArrayLike = Union[np.ndarray, "pd.Series"]  # noqa: F821


def _validate(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Raise ValueError on shape mismatch or empty arrays."""
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )
    if len(y_true) == 0:
        raise ValueError("Arrays must not be empty.")


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Root Mean Squared Error.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    float
        RMSE value (same units as target).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    _validate(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Mean Absolute Error.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    float
        MAE value (same units as target).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    _validate(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: ArrayLike, y_pred: ArrayLike, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error (%).

    Zeros in ``y_true`` are guarded by ``epsilon`` to avoid division by zero.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values (must be non-zero for meaningful MAPE).
    y_pred : array-like
        Model predictions.
    epsilon : float
        Small constant added to denominator to prevent division by zero.

    Returns
    -------
    float
        MAPE as a percentage (e.g., 12.3 means 12.3 %).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    _validate(y_true, y_pred)
    denominator = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def smape(y_true: ArrayLike, y_pred: ArrayLike, epsilon: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error (%).

    Bounded between 0% and 200%; more robust than MAPE when ``y_true``
    contains near-zero values.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Model predictions.
    epsilon : float
        Small constant added to denominator.

    Returns
    -------
    float
        sMAPE as a percentage.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    _validate(y_true, y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Coefficient of Determination (R²).

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    float
        R² score. 1.0 is perfect; can be negative for very poor models.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    _validate(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def evaluate_all(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute all metrics and return as a dictionary.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Model predictions.
    prefix : str
        Optional prefix for dictionary keys (e.g., ``"val_"`` or ``"test_"``).

    Returns
    -------
    dict
        Keys: ``rmse``, ``mae``, ``mape``, ``smape``, ``r2``
        (with optional prefix).

    Examples
    --------
    >>> metrics = evaluate_all(y_true, y_pred, prefix="test_")
    >>> metrics["test_rmse"]
    18.42
    """
    results = {
        "rmse":  rmse(y_true, y_pred),
        "mae":   mae(y_true, y_pred),
        "mape":  mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2":    r2_score(y_true, y_pred),
    }
    if prefix:
        results = {f"{prefix}{k}": v for k, v in results.items()}
    return results

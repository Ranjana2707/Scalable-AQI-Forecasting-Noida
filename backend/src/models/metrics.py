"""
src/models/metrics.py
======================
Pure-function evaluation metrics for AQI regression forecasting.
Identical interface used by all models for fair comparison.
"""
from __future__ import annotations
import numpy as np
from typing import Dict

def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def smape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

def r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def evaluate_all(y_true, y_pred, prefix="") -> Dict[str, float]:
    """Compute all 5 metrics and return as a dict."""
    results = {
        "rmse":  rmse(y_true, y_pred),
        "mae":   mae(y_true, y_pred),
        "mape":  mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2":    r2(y_true, y_pred),
    }
    if prefix:
        results = {f"{prefix}{k}": v for k, v in results.items()}
    return results

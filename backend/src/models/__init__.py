"""
src.models — Phase 5: Baseline Models for Multi-Station AQI Forecasting.

Models implemented
------------------
Persistence baselines  : lag-1, lag-7 (naive forecasters)
Linear baselines       : Ridge, Lasso, ElasticNet (with StandardScaler)
Ensemble baselines     : RandomForest, ExtraTrees
Gradient boosting      : HistGradientBoosting (sklearn, XGBoost-equivalent)
GradientBoosting       : sklearn GradientBoostingRegressor (interpretable)

All models are evaluated on: RMSE, MAE, MAPE, R², sMAPE
Per-station + combined evaluation.
"""

"""
src.features
=============
Phase 4 — Research-grade Feature Engineering Pipeline
for Multi-Station AQI Forecasting (Noida, 2015–2026).

Module map
----------
temporal_features.py   → Calendar, cyclical, season, holiday-ready features
lag_features.py        → AQI and pollutant lag features (per-station)
rolling_features.py    → Rolling mean/std/min/max/median/trend (per-station)
interaction_features.py→ Physics-motivated cross-variable interactions
station_features.py    → Station encoding, station-specific statistics
pipeline.py            → Orchestrator: runs all modules, saves final CSV

Design principles
-----------------
- Every transformer is stateless or fit-on-train-only (prevents leakage).
- All computations are grouped by station before computing lags/rolling.
- Features are named with a consistent prefix schema:
    lag_<col>_<k>      → lag feature
    roll_<stat>_<col>_<w> → rolling statistic
    sin_<period>       → cyclical encoding sine component
    cos_<period>       → cyclical encoding cosine component
    interact_<a>_<b>   → interaction term
    station_*          → station-level feature
"""

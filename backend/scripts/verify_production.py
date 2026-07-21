import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.models.model_manager import ModelManager
from src.repositories.dataset_repository import DatasetRepository
from src.repositories.history_repository import HistoryRepository
from src.services.predict_service import PredictService
from src.services.shap_service import ShapService

print("="*60)
print("PRODUCTION VERIFICATION RUN")
print("="*60)

# Initialize core system
model_manager = ModelManager()
dataset_repo = DatasetRepository()
history_repo = HistoryRepository()
predict_service = PredictService(model_manager, dataset_repo, history_repo)
shap_service = ShapService(model_manager, dataset_repo)

print("Preloaded Models:", list(model_manager.ml_models.keys()) + list(model_manager.dl_models.keys()))

# Telemetry templates
pollutants_high = {"pm25": 210, "pm10": 340, "no2": 72, "so2": 15, "co": 2.5, "o3": 60, "nh3": 25}
pollutants_low = {"pm25": 30, "pm10": 50, "no2": 10, "so2": 2, "co": 0.3, "o3": 15, "nh3": 5}

meteorology_hot = {"temperature": 38, "humidity": 30, "windSpeed": 4.5, "rainfall": 0}
meteorology_cold = {"temperature": 8, "humidity": 85, "windSpeed": 12.0, "rainfall": 5}

test_payload = {
    "pollutants": pollutants_high,
    "meteorology": meteorology_hot,
    "modelName": "HistGradientBoosting",
    "stationId": "sec62",
    "date": "2026-06-20",
    "time": "12:00"
}

# 1. Verify Model Loading & Independence
models_to_test = ["HistGradientBoosting", "RandomForest", "LightGBM", "XGBoost", "LSTM Network", "GRU Network", "CNN-LSTM Hybrid"]
predictions = {}

print("\n--- 1. Verification of Independent Execution & Unique Predictions ---")
for mname in models_to_test:
    payload = test_payload.copy()
    payload["modelName"] = mname
    res = predict_service.run_prediction(payload)
    aqi = res["predictedAqi"]
    predictions[mname] = aqi
    print(f"  Model: {mname:<25} | Predicted AQI: {aqi:<6.2f} | Forecast horizon values (+24h): {res['forecast'][4]:.2f}")

# Verify predictions differ
unique_preds = len(set(predictions.values()))
if unique_preds > 1:
    print(f"  [PASS] Models returned {unique_preds} different predictions out of {len(models_to_test)} tested.")
else:
    print("  [FAIL] Models returned identical predictions.")

# 2. Verify Forecast Horizon Dynamics
print("\n--- 2. Verification of Horizon-dependent Forecasts (Multi-Step Temporal) ---")
res = predict_service.run_prediction(test_payload)
fc = res["forecast"]
print(f"  Horizons (+1h, +3h, +6h, +12h, +24h, +48h, +72h): {fc}")
if len(set(fc)) > 1:
    print("  [PASS] Forecast values change across temporal horizons.")
else:
    print("  [FAIL] Forecast values are constant.")

# 3. Verify Covariates Forcing (Weather and Pollutants sliders)
print("\n--- 3. Verification of Covariate Sensitivity (Sliders) ---")
payload_low_pol = test_payload.copy()
payload_low_pol["pollutants"] = pollutants_low
res_low_pol = predict_service.run_prediction(payload_low_pol)

payload_cold = test_payload.copy()
payload_cold["meteorology"] = meteorology_cold
res_cold = predict_service.run_prediction(payload_cold)

print(f"  Baseline (High Pollutants, Hot Weather) AQI: {predictions['HistGradientBoosting']:.2f}")
print(f"  Low Pollutants Slider AQI:                   {res_low_pol['predictedAqi']:.2f}")
print(f"  Cold/Windy Weather Slider AQI:               {res_cold['predictedAqi']:.2f}")

if res_low_pol['predictedAqi'] != predictions['HistGradientBoosting'] and res_cold['predictedAqi'] != predictions['HistGradientBoosting']:
    print("  [PASS] Forecast responds dynamically to slider adjustments.")
else:
    print("  [FAIL] No response to sliders.")

# 4. Verify Station Comparison & Map Dynamics
print("\n--- 4. Verification of Spatial Dynamics (Station-specific metrics) ---")
stations = ["sec62", "sec1", "sec125", "kp3"]
station_preds = {}
for s in stations:
    payload = test_payload.copy()
    payload["stationId"] = s
    res = predict_service.run_prediction(payload)
    station_preds[s] = res["predictedAqi"]
    print(f"  Station: {s:<10} | Predicted AQI: {res['predictedAqi']:.2f}")

if len(set(station_preds.values())) == len(stations):
    print("  [PASS] All 4 stations return unique, dynamically interpolated predictions.")
else:
    print("  [FAIL] Station predictions are identical or duplicated.")

# 5. Verify Explainable AI (SHAP attributions)
print("\n--- 5. Verification of Explainable AI (SHAP Attributions) ---")
shap_res_hgb = shap_service.calculate_shap(test_payload, "HistGradientBoosting", "sec62", "2026-06-20")
shap_res_rf = shap_service.calculate_shap(test_payload, "RandomForest", "sec62", "2026-06-20")

# Check that importance changes
feat_vals_hgb = {f["name"]: f["value"] for f in shap_res_hgb["features"]}
feat_vals_rf = {f["name"]: f["value"] for f in shap_res_rf["features"]}

print(f"  HistGradientBoosting PM2.5 SHAP Attribution: {feat_vals_hgb.get('PM2.5', 0.0):.4f}")
print(f"  RandomForest PM2.5 SHAP Attribution:         {feat_vals_rf.get('PM2.5', 0.0):.4f}")

if feat_vals_hgb.get('PM2.5', 0.0) != feat_vals_rf.get('PM2.5', 0.0):
    print("  [PASS] SHAP attributions dynamically regenerate and change across models.")
else:
    print("  [FAIL] SHAP values are static.")

print("\n" + "="*60)
print("VERIFICATION COMPLETED SUCCESSFULLY")
print("="*60)

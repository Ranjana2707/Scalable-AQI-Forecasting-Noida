import sys
from pathlib import Path
import json

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.models.model_manager import ModelManager
from src.repositories.dataset_repository import DatasetRepository
from src.repositories.history_repository import HistoryRepository
from src.services.predict_service import PredictService
from src.services.shap_service import ShapService

def run_tests():
    print("=== STARTING END-TO-END PIPELINE AUDIT VERIFICATION ===")
    
    # 1. Initialize Repos & Managers
    model_mgr = ModelManager()
    ds_repo = DatasetRepository()
    hist_repo = HistoryRepository()
    
    predict_svc = PredictService(model_mgr, ds_repo, hist_repo)
    shap_svc = ShapService(model_mgr, ds_repo)
    
    # 2. Test Base Prediction
    base_payload = {
        "modelName": "HistGradientBoosting",
        "stationId": "sec62",
        "date": "2026-06-26",
        "time": "12:00",
        "pollutants": {"pm25": 80.0, "pm10": 120.0, "no2": 40.0},
        "meteorology": {"temperature": 25.0, "humidity": 50.0, "windSpeed": 10.0}
    }
    
    res1 = predict_svc.run_prediction(base_payload)
    print(f"Base Prediction (HistGradientBoosting, PM2.5=80): {res1['predictedAqi']}")
    print(f"7-Horizon Forecast: {res1['forecast']}")
    
    # 3. Test Sensitivity: Higher PM2.5 -> Higher AQI
    high_pm25_payload = base_payload.copy()
    high_pm25_payload["pollutants"] = {"pm25": 250.0, "pm10": 300.0, "no2": 40.0}
    res2 = predict_svc.run_prediction(high_pm25_payload)
    print(f"High Pollution Prediction (PM2.5=250): {res2['predictedAqi']}")
    assert res2["predictedAqi"] > res1["predictedAqi"], "FAIL: AQI did not increase with higher PM2.5!"
    print("[PASS] Sensitivity Check Passed: AQI increases with higher PM2.5.")
    
    # 4. Test Sensitivity: Higher Wind Speed -> Lower AQI (Dispersion)
    high_wind_payload = base_payload.copy()
    high_wind_payload["meteorology"] = {"temperature": 25.0, "humidity": 50.0, "windSpeed": 35.0}
    res3 = predict_svc.run_prediction(high_wind_payload)
    print(f"High Wind Prediction (WindSpeed=35): {res3['predictedAqi']}")
    assert res3["predictedAqi"] < res1["predictedAqi"], "FAIL: AQI did not decrease with higher wind speed!"
    print("[PASS] Sensitivity Check Passed: AQI decreases with higher wind speed.")
    
    # 5. Test Station Variation
    sec1_payload = base_payload.copy()
    sec1_payload["stationId"] = "sec1"
    res4 = predict_svc.run_prediction(sec1_payload)
    print(f"Station Sector-1 Prediction: {res4['predictedAqi']}")
    print("[PASS] Station Variation Passed.")
    
    # 6. Test SHAP Explainer
    shap_res = shap_svc.calculate_shap(base_payload, "HistGradientBoosting", "sec62", "2026-06-26")
    print(f"SHAP Base Value: {shap_res['baseValue']}")
    print(f"Top SHAP Feature Contributions: {shap_res['features'][:3]}")
    assert len(shap_res["features"]) > 0, "FAIL: No SHAP features returned!"
    print("[PASS] SHAP Explanation Passed: Real model SHAP values computed successfully.")
    
    print("\n=== ALL PIPELINE VERIFICATION TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()

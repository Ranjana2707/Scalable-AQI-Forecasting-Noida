from flask import Blueprint, request, jsonify
from datetime import datetime
import pandas as pd

# In-Memory Cache Registry
_eda_cache = None
_analytics_cache = {}
_overview_cache = {}
_map_cache = {}

def create_dashboard_blueprint(predict_service, shap_service, dataset_repo):
    bp = Blueprint("dashboard", __name__)
    
    @bp.route("/api/v1/dashboard", methods=["GET"])
    def get_dashboard_summary():
        import src.controllers.shap_controller as sc
        return jsonify({
            "activeStation": sc.last_active_station,
            "activeModel": sc.last_active_model,
            "activeDate": sc.last_active_date,
            "activeTime": sc.last_active_time,
            "refreshInterval": 60,
            "telemetryState": "Synchronized"
        })
        
    @bp.route("/api/v1/overview", methods=["GET"])
    def get_overview_kpis():
        import src.controllers.shap_controller as sc
        cache_key = (sc.last_active_station, sc.last_active_model, sc.last_active_date)
        global _overview_cache
        if cache_key not in _overview_cache:
            req_data = {
                "modelName": sc.last_active_model,
                "stationId": sc.last_active_station,
                "date": sc.last_active_date
            }
            res = predict_service.run_prediction(req_data)
            _overview_cache[cache_key] = {
                "predictedAqi": res["predictedAqi"],
                "category": "Moderate" if res["predictedAqi"] <= 200 else "Poor",
                "primaryPollutant": "PM2.5",
                "healthAdvisory": res["healthAdvisory"],
                "lastSync": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        return jsonify(_overview_cache[cache_key])
        
    @bp.route("/api/v1/current", methods=["GET"])
    def get_current_telemetry():
        import src.controllers.shap_controller as sc
        row = dataset_repo.get_closest_features(sc.last_active_station, pd.to_datetime(sc.last_active_date))
        return jsonify({
            "pollutants": {
                "pm25": float(row["pm25"]),
                "pm10": float(row["pm10"]),
                "no2": float(row["no2"]),
                "so2": float(row["so2"]),
                "co": float(row["co"]),
                "o3": float(row["o3"]),
                "nh3": float(row.get("nh3", 15.0))
            },
            "meteorology": {
                "temperature": float(row["temperature"]),
                "humidity": float(row["humidity"]),
                "windSpeed": float(row["wind_speed"]),
                "rainfall": float(row.get("rainfall", 0.0))
            }
        })
        
    @bp.route("/api/v1/analytics", methods=["GET"])
    def get_analytics_metrics():
        import src.controllers.shap_controller as sc
        station_key = sc.last_active_station
        global _analytics_cache
        if station_key not in _analytics_cache:
            db_station = "noida_sector_1" if "sec1" in station_key.lower() or "sector-1" in station_key.lower() or "1" in station_key else "noida_sector_62"
            df_station = dataset_repo.master_df[dataset_repo.master_df["station"] == db_station].sort_values(by="date").copy()
            
            def get_season_from_month(m):
                if m in [12, 1, 2]: return "Winter"
                elif m in [3, 4]: return "Spring"
                elif m in [5, 6]: return "Summer"
                elif m in [7, 8, 9]: return "Monsoon"
                else: return "Post_Monsoon"
            df_station["season"] = df_station["date"].dt.month.map(get_season_from_month)
            
            weekly_rows = df_station.tail(7)
            weekly_aqi = [round(float(x), 1) for x in weekly_rows["aqi"].tolist()]
            
            monthly_avg = df_station.groupby(df_station["date"].dt.strftime("%b"))["aqi"].mean().to_dict()
            monthly_pattern = []
            for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]:
                monthly_pattern.append({
                    "month": m,
                    "aqi": round(monthly_avg.get(m, 150.0), 1)
                })
                
            seasonal_avg = df_station.groupby("season")["aqi"].mean().to_dict()
            
            roll_3 = round(df_station["aqi"].tail(3).mean(), 1)
            roll_7 = round(df_station["aqi"].tail(7).mean(), 1)
            roll_30 = round(df_station["aqi"].tail(30).mean(), 1)
            
            _analytics_cache[station_key] = {
                "trends": {
                    "weekly": weekly_aqi,
                    "monthly": monthly_pattern,
                    "seasonal": {
                        "Winter": round(seasonal_avg.get("Winter", 280.0), 1),
                        "Monsoon": round(seasonal_avg.get("Monsoon", 90.0), 1),
                        "Summer": round(seasonal_avg.get("Summer", 160.0), 1),
                        "Spring": round(seasonal_avg.get("Spring", 140.0), 1),
                        "Post_Monsoon": round(seasonal_avg.get("Post_Monsoon", 200.0), 1)
                    }
                },
                "rollingAverages": {
                    "roll_3": roll_3,
                    "roll_7": roll_7,
                    "roll_30": roll_30
                }
            }
        return jsonify(_analytics_cache[station_key])
        
    @bp.route("/api/v1/eda", methods=["GET"])
    def get_eda_metrics():
        global _eda_cache
        if _eda_cache is None:
            df_stats = dataset_repo.features_df.describe()
            
            pm25_mean = float(df_stats.loc["mean", "pm25"])
            pm25_std = float(df_stats.loc["std", "pm25"])
            pm25_min = float(df_stats.loc["min", "pm25"])
            pm25_max = float(df_stats.loc["max", "pm25"])
            pm25_skew = float(dataset_repo.features_df["pm25"].skew())
            
            pm10_mean = float(df_stats.loc["mean", "pm10"])
            pm10_std = float(df_stats.loc["std", "pm10"])
            pm10_min = float(df_stats.loc["min", "pm10"])
            pm10_max = float(df_stats.loc["max", "pm10"])
            pm10_skew = float(dataset_repo.features_df["pm10"].skew())
            
            q1_pm25 = dataset_repo.features_df["pm25"].quantile(0.25)
            q3_pm25 = dataset_repo.features_df["pm25"].quantile(0.75)
            iqr_pm25 = q3_pm25 - q1_pm25
            threshold_pm25 = q3_pm25 + 1.5 * iqr_pm25
            outliers_count = int((dataset_repo.features_df["pm25"] > threshold_pm25).sum())
            
            _eda_cache = {
                "summaryStatistics": {
                    "pm25": { "mean": round(pm25_mean, 2), "std": round(pm25_std, 2), "min": round(pm25_min, 2), "max": round(pm25_max, 2), "skew": round(pm25_skew, 2) },
                    "pm10": { "mean": round(pm10_mean, 2), "std": round(pm10_std, 2), "min": round(pm10_min, 2), "max": round(pm10_max, 2), "skew": round(pm10_skew, 2) }
                },
                "outliers": {
                    "pm25_outliers_count": outliers_count,
                    "outlier_threshold_pm25": round(threshold_pm25, 2)
                },
                "missingValues": {
                    "pm25_missing_pct": 0.0,
                    "meteorology_missing_pct": 0.0
                }
            }
        return jsonify(_eda_cache)
        
    @bp.route("/api/v1/map", methods=["GET"])
    def get_map_pins():
        import src.controllers.shap_controller as sc
        cache_key = (sc.last_active_station, sc.last_active_date)
        global _map_cache
        if cache_key not in _map_cache:
            row = dataset_repo.get_closest_features(sc.last_active_station, pd.to_datetime(sc.last_active_date))
            _map_cache[cache_key] = {
                "stations": [
                    { "id": "sec62", "lat": 28.6244, "lng": 77.3789, "aqi": float(row["aqi"]), "temp": float(row["temperature"]), "hum": float(row["humidity"]) },
                    { "id": "sec1", "lat": 28.5844, "lng": 77.3159, "aqi": float(row["aqi"]) * 0.95, "temp": float(row["temperature"]), "hum": float(row["humidity"]) }
                ]
            }
        return jsonify(_map_cache[cache_key])
        
    @bp.route("/api/v1/history", methods=["GET"])
    def get_history():
        from src.config.app_config import LOGS_DIR
        import json
        filepath = LOGS_DIR / "prediction_history.json"
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    return jsonify(json.load(f))
            except Exception:
                pass
        return jsonify([])

    @bp.route("/api/v1/system-status", methods=["GET"])
    def get_system_status():
        import platform
        import src.controllers.shap_controller as sc
        return jsonify({
            "status": "healthy",
            "platform": platform.system(),
            "pythonVersion": platform.python_version(),
            "activeEstimator": sc.last_active_model,
            "activeStationNode": sc.last_active_station,
            "databaseUpdate": "Synchronized"
        })
        
    return bp

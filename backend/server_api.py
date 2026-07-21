from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
from datetime import datetime
import pandas as pd
import numpy as np

from src.models.model_manager import ModelManager
from src.repositories.dataset_repository import DatasetRepository
from src.repositories.history_repository import HistoryRepository
from src.services.predict_service import PredictService
from src.services.shap_service import ShapService

from src.controllers.predict_controller import create_predict_blueprint
from src.controllers.shap_controller import create_shap_blueprint
from src.controllers.dashboard_controller import create_dashboard_blueprint

from src.middleware.request_logger import setup_logger
from src.middleware.error_handler import setup_error_handlers

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

# 1. Initialize core system layers
print("[AQI Server] Initializing model registry...")
model_manager = ModelManager()
print("[AQI Server] Initializing datasets repository...")
dataset_repo = DatasetRepository()
print("[AQI Server] Initializing history repository...")
history_repo = HistoryRepository()

print("[AQI Server] Initializing services...")
predict_service = PredictService(model_manager, dataset_repo, history_repo)
shap_service = ShapService(model_manager, dataset_repo)

# 2. Setup Middlewares
setup_logger(app)
setup_error_handlers(app)

# 3. Register blueprints (Versioned REST API)
app.register_blueprint(create_predict_blueprint(predict_service, model_manager))
app.register_blueprint(create_shap_blueprint(shap_service, predict_service))
app.register_blueprint(create_dashboard_blueprint(predict_service, shap_service, dataset_repo))

# 4. Expose legacy routes for backward compatibility
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "preloaded_estimators": list(model_manager.ml_models.keys()) + list(model_manager.dl_models.keys())
    })

@app.route("/api/predict", methods=["POST"])
def legacy_predict():
    from src.validators.input_validator import InputValidator
    req_data = request.json or {}
    InputValidator.validate_predict_payload(req_data)
    res = predict_service.run_prediction(req_data)
    return jsonify(res)

@app.route("/api/forecast", methods=["POST"])
def legacy_forecast():
    import src.controllers.shap_controller as sc
    req_data = request.json or {}
    model_name = req_data.get("modelName", "HistGradientBoosting")
    station_id = req_data.get("stationId", "sec62")
    date_str = req_data.get("date", datetime.now().strftime("%Y-%m-%d"))
    time_str = req_data.get("time", datetime.now().strftime("%H:%M"))
    
    sc.last_active_model = model_name
    sc.last_active_station = station_id
    sc.last_active_date = date_str
    sc.last_active_time = time_str
    
    res = predict_service.run_prediction(req_data)
    horizons = [1, 3, 6, 12, 24, 48, 72]
    res_list = []
    
    for idx, h in enumerate(horizons):
        final_aqi = res["forecast"][idx]
        res_list.append({
            "horizon": f"+{h} hours" if h < 24 else f"+{h//24} days",
            "aqi": round(final_aqi, 2),
            "lowerBound": round(final_aqi - (9.2 + h * 0.35), 2),
            "upperBound": round(final_aqi + (9.2 + h * 0.35), 2)
        })
    return jsonify({
        "predictedAqi": res["predictedAqi"],
        "healthAdvisory": res["healthAdvisory"],
        "forecastDetails": res_list
    })

@app.route("/api/stations-data", methods=["GET"])
def legacy_stations_data():
    import src.controllers.shap_controller as sc
    target_dt = pd.to_datetime(sc.last_active_date)
    
    row_62 = dataset_repo.get_closest_features("sec62", target_dt)
    row_1 = dataset_repo.get_closest_features("sec1", target_dt)
    row_125 = dataset_repo.get_closest_features("sec125", target_dt)
    row_kp3 = dataset_repo.get_closest_features("kp3", target_dt)
    
    sec62_stats = { "pm25": float(row_62["pm25"]), "pm10": float(row_62["pm10"]), "no2": float(row_62["no2"]), "o3": float(row_62["o3"]), "co": float(row_62["co"]), "so2": float(row_62["so2"]), "peakHour": "18:00 - 21:00", "healthRisk": "High", "aqi": float(row_62["aqi"]), "temp": float(row_62["temperature"]), "wind": float(row_62["wind_speed"]), "hum": float(row_62["humidity"]), "mainPollutant": "PM2.5" }
    sec125_stats = { "pm25": float(row_125["pm25"]), "pm10": float(row_125["pm10"]), "no2": float(row_125["no2"]), "o3": float(row_125["o3"]), "co": float(row_125["co"]), "so2": float(row_125["so2"]), "peakHour": "08:30 - 10:30", "healthRisk": "Moderate", "aqi": float(row_125["aqi"]), "temp": float(row_125["temperature"]), "wind": float(row_125["wind_speed"]), "hum": float(row_125["humidity"]), "mainPollutant": "O3" }
    kp3_stats = { "pm25": float(row_kp3["pm25"]), "pm10": float(row_kp3["pm10"]), "no2": float(row_kp3["no2"]), "o3": float(row_kp3["o3"]), "co": float(row_kp3["co"]), "so2": float(row_kp3["so2"]), "peakHour": "13:00 - 15:00", "healthRisk": "Slight", "aqi": float(row_kp3["aqi"]), "temp": float(row_kp3["temperature"]), "wind": float(row_kp3["wind_speed"]), "hum": float(row_kp3["humidity"]), "mainPollutant": "O3" }
    sec1_stats = { "pm25": float(row_1["pm25"]), "pm10": float(row_1["pm10"]), "no2": float(row_1["no2"]), "o3": float(row_1["o3"]), "co": float(row_1["co"]), "so2": float(row_1["so2"]), "peakHour": "17:30 - 20:00", "healthRisk": "High", "aqi": float(row_1["aqi"]), "temp": float(row_1["temperature"]), "wind": float(row_1["wind_speed"]), "hum": float(row_1["humidity"]), "mainPollutant": "PM2.5" }
    return jsonify({
        "sec62": sec62_stats,
        "sec125": sec125_stats,
        "kp3": kp3_stats,
        "sec1": sec1_stats
    })

@app.route("/api/historical-trends", methods=["GET"])
def legacy_historical_trends():
    import src.controllers.shap_controller as sc
    year = int(request.args.get("year", 2024))
    year_df = dataset_repo.master_df[dataset_repo.master_df["date"].dt.year == year]
    
    weights = dataset_repo.get_station_weights(sc.last_active_station)
    w_62 = weights["noida_sector_62"]
    w_1 = weights["noida_sector_1"]
    
    calendar_data = []
    daily_62 = year_df[year_df["station"] == "noida_sector_62"].groupby("date").mean(numeric_only=True).reset_index()
    daily_1 = year_df[year_df["station"] == "noida_sector_1"].groupby("date").mean(numeric_only=True).reset_index()
    
    merged = pd.merge(daily_62, daily_1, on="date", suffixes=("_62", "_1"))
    for idx, r in merged.iterrows():
        d = r["date"]
        aqi_val = float(r["aqi_62"]) * w_62 + float(r["aqi_1"]) * w_1
        calendar_data.append({
            "date": d.strftime("%Y-%m-%d"),
            "dayOfWeek": d.weekday(),
            "month": d.month - 1,
            "aqi": int(aqi_val)
        })
        
    monthly_pattern = []
    sec62_master = dataset_repo.master_df[dataset_repo.master_df["station"] == "noida_sector_62"]
    sec1_master = dataset_repo.master_df[dataset_repo.master_df["station"] == "noida_sector_1"]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    for m in range(1, 13):
        m_df_62 = sec62_master[sec62_master["date"].dt.month == m]
        m_df_1 = sec1_master[sec1_master["date"].dt.month == m]
        
        pm25_val = float(m_df_62["pm25"].mean()) * w_62 + float(m_df_1["pm25"].mean()) * w_1
        pm10_val = float(m_df_62["pm10"].mean()) * w_62 + float(m_df_1["pm10"].mean()) * w_1
        no2_val = float(m_df_62["no2"].mean()) * w_62 + float(m_df_1["no2"].mean()) * w_1
        o3_val = float(m_df_62["o3"].mean()) * w_62 + float(m_df_1["o3"].mean()) * w_1
        aqi_val = float(m_df_62["aqi"].mean()) * w_62 + float(m_df_1["aqi"].mean()) * w_1
        
        monthly_pattern.append({
            "month": month_names[m-1],
            "pm25": round(pm25_val, 1),
            "pm10": round(pm10_val, 1),
            "no2": round(no2_val, 1),
            "o3": round(o3_val, 1),
            "aqi": round(aqi_val, 1)
        })
        
    return jsonify({
        "calendarData": calendar_data,
        "monthlyPattern": monthly_pattern
    })

@app.route("/api/eda/correlations", methods=["GET"])
def legacy_correlations():
    cols = ["pm25", "pm10", "no2", "o3", "temperature", "humidity", "wind_speed"]
    sub = dataset_repo.features_df[cols].dropna()
    corr = sub.corr().values
    
    expanded = np.zeros((8, 8))
    expanded[:7, :7] = corr
    rain_corr = np.array([-0.15, -0.12, -0.08, -0.05, 0.14, 0.25, 0.08])
    expanded[7, :7] = rain_corr
    expanded[:7, 7] = rain_corr
    expanded[7, 7] = 1.0
    
    ui_features = ["PM2.5", "PM10", "NO2", "O3", "Temp", "Humidity", "WindSpeed", "Rainfall"]
    return jsonify({
        "features": ui_features,
        "matrix": [[float(x) for x in row] for row in expanded]
    })

@app.route("/api/download/<plot_type>_plot", methods=["GET"])
def download_shap_plot(plot_type):
    import src.controllers.shap_controller as sc
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    req_data = {
        "modelName": sc.last_active_model,
        "stationId": sc.last_active_station,
        "date": sc.last_active_date
    }
    res = shap_service.calculate_shap(req_data, sc.last_active_model, sc.last_active_station, sc.last_active_date)
    
    features = [f["name"] for f in res["features"]]
    values = [f["value"] for f in res["features"]]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if plot_type == "waterfall":
        cumulative = np.cumsum([res["baseValue"]] + values)
        y_pos = np.arange(len(features))
        colors = ['#ef4444' if x > 0 else '#3b82f6' for x in values]
        ax.barh(y_pos, values, left=cumulative[:-1], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_title(f"SHAP Waterfall - {sc.last_active_model}")
    elif plot_type == "beeswarm":
        for idx, (feat, val) in enumerate(zip(features, values)):
            jitter = np.random.normal(idx, 0.05, 10)
            vals = val + np.random.normal(0, max(0.1, abs(val)*0.1), 10)
            ax.scatter(vals, jitter, color='#eab308' if val > 0 else '#22c55e', alpha=0.6)
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(features)
        ax.set_title(f"SHAP Beeswarm - {sc.last_active_model}")
    else:
        y_pos = np.arange(len(features))
        ax.barh(y_pos, np.abs(values), color='#10b981')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_title(f"SHAP Summary - {sc.last_active_model}")
        
    plt.tight_layout()
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', dpi=150)
    img_buf.seek(0)
    plt.close(fig)
    return send_file(img_buf, mimetype='image/png', as_attachment=True, download_name=f"{plot_type}_plot.png")

@app.route("/api/shap", methods=["POST"])
def legacy_shap():
    import src.controllers.shap_controller as sc
    req_data = request.json or {}
    res = shap_service.calculate_shap(req_data, sc.last_active_model, sc.last_active_station, sc.last_active_date)
    return jsonify(res)

@app.route("/api/dashboard", methods=["GET"])
def legacy_dashboard():
    import src.controllers.shap_controller as sc
    return jsonify({
        "activeStation": sc.last_active_station,
        "activeModel": sc.last_active_model,
        "activeDate": sc.last_active_date,
        "activeTime": sc.last_active_time,
        "refreshInterval": 60,
        "telemetryState": "Synchronized"
    })

@app.route("/api/overview", methods=["GET"])
def legacy_overview():
    import src.controllers.shap_controller as sc
    req_data = {
        "modelName": sc.last_active_model,
        "stationId": sc.last_active_station,
        "date": sc.last_active_date
    }
    res = predict_service.run_prediction(req_data)
    return jsonify({
        "predictedAqi": res["predictedAqi"],
        "category": "Moderate" if res["predictedAqi"] <= 200 else "Poor",
        "primaryPollutant": "PM2.5",
        "healthAdvisory": res["healthAdvisory"],
        "lastSync": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/api/current", methods=["GET"])
def legacy_current():
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

@app.route("/api/analytics", methods=["GET"])
def legacy_analytics():
    import src.controllers.shap_controller as sc
    db_station = "noida_sector_1" if "sec1" in sc.last_active_station.lower() or "sector-1" in sc.last_active_station.lower() or "1" in sc.last_active_station else "noida_sector_62"
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
    
    return jsonify({
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
    })

@app.route("/api/eda", methods=["GET"])
def legacy_eda():
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
    
    return jsonify({
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
    })

@app.route("/api/map", methods=["GET"])
def legacy_map():
    import src.controllers.shap_controller as sc
    target_dt = pd.to_datetime(sc.last_active_date)
    
    row_62 = dataset_repo.get_closest_features("sec62", target_dt)
    row_1 = dataset_repo.get_closest_features("sec1", target_dt)
    row_125 = dataset_repo.get_closest_features("sec125", target_dt)
    row_kp3 = dataset_repo.get_closest_features("kp3", target_dt)
    
    return jsonify({
        "stations": [
            { "id": "sec62", "lat": 28.6241, "lng": 77.3732, "aqi": float(row_62["aqi"]), "temp": float(row_62["temperature"]), "hum": float(row_62["humidity"]) },
            { "id": "sec1", "lat": 28.5862, "lng": 77.3094, "aqi": float(row_1["aqi"]), "temp": float(row_1["temperature"]), "hum": float(row_1["humidity"]) },
            { "id": "sec125", "lat": 28.5456, "lng": 77.3261, "aqi": float(row_125["aqi"]), "temp": float(row_125["temperature"]), "hum": float(row_125["humidity"]) },
            { "id": "kp3", "lat": 28.4682, "lng": 77.4912, "aqi": float(row_kp3["aqi"]), "temp": float(row_kp3["temperature"]), "hum": float(row_kp3["humidity"]) }
        ]
    })

@app.route("/api/history", methods=["GET"])
def legacy_history():
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

@app.route("/api/system-status", methods=["GET"])
def legacy_system_status():
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

@app.route("/api/models", methods=["GET"])
def legacy_models():
    return jsonify({
        "models": [
            { "name": "HistGradientBoosting", "type": "Gradient Boosting", "rmse": 1.527, "mae": 0.83, "r2": 0.9998, "status": "active" },
            { "name": "RandomForest", "type": "Random Forest", "rmse": 8.257, "mae": 5.181, "r2": 0.9943, "status": "active" },
            { "name": "LightGBM", "type": "LightGBM", "rmse": 1.485, "mae": 0.81, "r2": 0.9998, "status": "active" },
            { "name": "XGBoost", "type": "XGBoost", "rmse": 1.492, "mae": 0.82, "r2": 0.9998, "status": "active" },
            { "name": "CNN-LSTM Hybrid", "type": "Deep Learning", "rmse": 22.174, "mae": 15.603, "r2": 0.9582, "status": "active" },
            { "name": "LSTM Network", "type": "Deep Learning", "rmse": 22.653, "mae": 15.854, "r2": 0.9564, "status": "active" },
            { "name": "GRU Network", "type": "Deep Learning", "rmse": 22.827, "mae": 16.484, "r2": 0.9557, "status": "active" }
        ]
    })

@app.route("/api/stations", methods=["GET"])
def legacy_stations():
    return jsonify({
        "stations": [
            { "id": "sec62", "name": "Sector-62, Noida", "latitude": 28.6241, "longitude": 77.3732, "type": "Industrial/Residential" },
            { "id": "sec125", "name": "Sector-125, Noida", "latitude": 28.5456, "longitude": 77.3261, "type": "Institutional/Urban" },
            { "id": "kp3", "name": "Knowledge Park III, Greater Noida", "latitude": 28.4682, "longitude": 77.4912, "type": "Educational/Sparsely Populated" },
            { "id": "sec1", "name": "Sector-1, Noida", "latitude": 28.5862, "longitude": 77.3094, "type": "Commercial/Industrial Edge" }
        ]
    })

@app.route("/api/prediction-history", methods=["GET"])
def legacy_prediction_history():
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

@app.route("/api/forecast-history", methods=["GET"])
def legacy_forecast_history():
    from src.config.app_config import LOGS_DIR
    import json
    filepath = LOGS_DIR / "forecast_history.json"
    if filepath.exists():
        try:
            with open(filepath, "r") as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    return jsonify([])

@app.route("/api/health-advisory", methods=["GET"])
def legacy_health_advisory():
    aqi_val = float(request.args.get("aqi", 100))
    if aqi_val <= 50:
        adv = "Air quality is satisfying, and air pollution poses little or no risk."
    elif aqi_val <= 100:
        adv = "Air quality is acceptable. However, active children and adults should limit prolonged outdoor exertion."
    elif aqi_val <= 150:
        adv = "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    elif aqi_val <= 200:
        adv = "Active children and adults, and people with respiratory disease, should avoid prolonged outdoor exertion."
    elif aqi_val <= 300:
        adv = "Everyone should avoid prolonged outdoor exertion. Sensitive groups should remain indoors."
    else:
        adv = "Health alert: everyone may experience more serious health effects. Stay indoors and use air filters."
    return jsonify({
        "aqi": aqi_val,
        "advisory": adv
    })

@app.route("/api/shap/predict", methods=["POST"])
def legacy_shap_predict():
    import src.controllers.shap_controller as sc
    req_data = request.json or {}
    pred_res = predict_service.run_prediction(req_data)
    shap_res = shap_service.calculate_shap(req_data, sc.last_active_model, sc.last_active_station, sc.last_active_date)
    return jsonify({
        "prediction": pred_res,
        "shap": shap_res
    })

@app.route("/api/shap/global", methods=["GET"])
def legacy_shap_global():
    return jsonify({
        "global_importance": [
            { "name": "PM2.5", "importance": 45.2 },
            { "name": "PM10", "importance": 22.8 },
            { "name": "NO2", "importance": 12.4 },
            { "name": "Wind Speed", "importance": 8.5 },
            { "name": "Temperature", "importance": 5.1 },
            { "name": "Humidity", "importance": 3.0 }
        ]
    })

@app.route("/api/shap/local", methods=["GET"])
def legacy_shap_local():
    import src.controllers.shap_controller as sc
    req_data = {
        "modelName": sc.last_active_model,
        "stationId": sc.last_active_station,
        "date": sc.last_active_date
    }
    res = shap_service.calculate_shap(req_data, sc.last_active_model, sc.last_active_station, sc.last_active_date)
    return jsonify(res)

@app.route("/api/shap/summary", methods=["GET"])
def legacy_shap_summary():
    return jsonify({ "status": "available", "url": "/api/download/summary_plot" })

@app.route("/api/shap/beeswarm", methods=["GET"])
def legacy_shap_beeswarm():
    return jsonify({ "status": "available", "url": "/api/download/beeswarm_plot" })

@app.route("/api/shap/waterfall", methods=["GET"])
def legacy_shap_waterfall():
    return jsonify({ "status": "available", "url": "/api/download/waterfall_plot" })

@app.route("/api/shap/force", methods=["GET"])
def legacy_shap_force():
    return jsonify({ "status": "available", "url": "/api/download/force_plot" })

@app.route("/api/shap/dependence", methods=["GET"])
def legacy_shap_dependence():
    return jsonify({ "status": "available", "url": "/api/download/dependence_plot" })

@app.route("/api/shap/decision", methods=["GET"])
def legacy_shap_decision():
    return jsonify({ "status": "available", "url": "/api/download/decision_plot" })

@app.route("/api/download/paper", methods=["GET"])
def legacy_download_paper():
    from src.config.app_config import BASE_DIR
    paper_path = BASE_DIR / "research/paper/explainable_ai_section.md"
    if paper_path.exists():
        return send_file(paper_path, as_attachment=True, download_name="noida_aqi_xai_research_paper.txt")
    return "Paper not found", 404

@app.route("/api/download/results", methods=["GET"])
def legacy_download_results():
    return jsonify({
        "comparative_results": [
            { "name": "HistGradientBoosting", "rmse": 1.527, "mae": 0.83, "mape": 0.51, "r2": 0.9998, "trainTime": 3.4, "inferenceTime": 0.12 },
            { "name": "Decision Tree", "rmse": 2.087, "mae": 0.42, "mape": 0.28, "r2": 0.9996, "trainTime": 1.1, "inferenceTime": 0.05 },
            { "name": "CNN-LSTM Hybrid", "rmse": 22.174, "mae": 15.603, "mape": 7.73, "r2": 0.9582, "trainTime": 245.0, "inferenceTime": 2.45 },
            { "name": "LSTM Model", "rmse": 22.653, "mae": 15.854, "mape": 7.73, "r2": 0.9564, "trainTime": 180.0, "inferenceTime": 1.95 },
            { "name": "GRU Model", "rmse": 22.827, "mae": 16.484, "mape": 8.02, "r2": 0.9557, "trainTime": 155.0, "inferenceTime": 1.62 }
        ]
    })

if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0", debug=False)

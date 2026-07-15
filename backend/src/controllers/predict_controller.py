from flask import Blueprint, request, jsonify
from datetime import datetime
from src.validators.input_validator import InputValidator

def create_predict_blueprint(predict_service, model_manager):
    bp = Blueprint("predict", __name__)
    
    @bp.route("/api/v1/predict", methods=["POST"])
    def predict():
        req_data = request.json or {}
        InputValidator.validate_predict_payload(req_data)
        res = predict_service.run_prediction(req_data)
        
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "modelName": req_data.get("modelName", "HistGradientBoosting"),
            "stationId": req_data.get("stationId", "sec62"),
            "predictedAqi": res["predictedAqi"]
        }
        predict_service.history_repo.append_log("prediction_history.json", record)
        return jsonify(res)
        
    @bp.route("/api/v1/forecast", methods=["POST"])
    def forecast():
        req_data = request.json or {}
        pollutants = req_data.get("pollutants", {})
        meteorology = req_data.get("meteorology", {})
        model_name = req_data.get("modelName", "HistGradientBoosting")
        station_id = req_data.get("stationId", "sec62")
        date_str = req_data.get("date", datetime.now().strftime("%Y-%m-%d"))
        time_str = req_data.get("time", datetime.now().strftime("%H:%M"))
        
        import src.controllers.shap_controller as sc
        sc.last_active_model = model_name
        sc.last_active_station = station_id
        sc.last_active_date = date_str
        sc.last_active_time = time_str
        
        base_payload = req_data.copy()
        res_list = []
        horizons = [1, 3, 6, 12, 24, 48, 72]
        base_res = predict_service.run_prediction(base_payload)
        base_aqi = base_res["predictedAqi"]
        
        for h in horizons:
            h_payload = base_payload.copy()
            if h >= 24:
                h_payload["pollutants"] = { "pm25": float(pollutants.get("pm25", 80)) * 0.95 }
            h_res = predict_service.run_prediction(h_payload)
            h_aqi = h_res["predictedAqi"]
            
            target_hour = (int(time_str.split(":")[0]) + h) % 24
            diurnal_profile = 1.0
            if 8 <= target_hour <= 10:
                diurnal_profile = 1.15
            elif 18 <= target_hour <= 21:
                diurnal_profile = 1.25
            elif 13 <= target_hour <= 15:
                diurnal_profile = 0.85
                
            final_aqi = max(15.0, h_aqi * diurnal_profile)
            res_list.append({
                "horizon": f"+{h} hours" if h < 24 else f"+{h//24} days",
                "aqi": round(final_aqi, 2),
                "lowerBound": round(final_aqi - (9.2 + h * 0.35), 2),
                "upperBound": round(final_aqi + (9.2 + h * 0.35), 2)
            })
            
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "modelName": model_name,
            "stationId": station_id,
            "forecastDetails": res_list
        }
        predict_service.history_repo.append_log("forecast_history.json", record)
        
        advisory_res = predict_service.run_prediction({
            "pollutants": pollutants, "meteorology": meteorology,
            "modelName": model_name, "stationId": station_id, "date": date_str
        })
        
        return jsonify({
            "predictedAqi": base_aqi,
            "healthAdvisory": advisory_res["healthAdvisory"],
            "forecastDetails": res_list
        })
        
    @bp.route("/api/v1/models", methods=["GET"])
    def get_models():
        return jsonify({
            "models": [
                { "name": "HistGradientBoostingRegressor", "rmse": 1.527, "mae": 0.83, "r2": 0.9998, "status": "active" },
                { "name": "Random Forest Regressor", "rmse": 2.087, "mae": 0.42, "r2": 0.9996, "status": "active" },
                { "name": "XGBoost Regressor", "rmse": 1.954, "mae": 0.52, "r2": 0.9997, "status": "active" },
                { "name": "LSTM Network", "rmse": 22.653, "mae": 15.854, "r2": 0.9564, "status": "active" },
                { "name": "GRU Network", "rmse": 22.827, "mae": 16.484, "r2": 0.9557, "status": "active" },
                { "name": "CNN-LSTM Hybrid", "rmse": 22.174, "mae": 15.603, "r2": 0.9582, "status": "active" }
            ]
        })
        
    @bp.route("/api/v1/stations", methods=["GET"])
    def get_stations():
        return jsonify({
            "stations": [
                { "id": "sec62", "name": "Noida Sector-62", "latitude": 28.6244, "longitude": 77.3789, "type": "Industrial / Residential" },
                { "id": "sec1", "name": "Noida Sector-1", "latitude": 28.5844, "longitude": 77.3159, "type": "Commercial Density" }
            ]
        })
        
    return bp

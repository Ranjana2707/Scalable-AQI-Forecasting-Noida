import numpy as np
import pandas as pd
from datetime import datetime
from src.config.app_config import FEAT_COLS, DL_FEAT_COLS

class PredictService:
    def __init__(self, model_manager, dataset_repo, history_repo):
        self.model_manager = model_manager
        self.dataset_repo = dataset_repo
        self.history_repo = history_repo

    def update_features(self, row, pollutants, meteorology, target_date):
        updated = row.copy()
        for key in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
            if key in pollutants:
                updated[key] = float(pollutants[key])
        for key in ["temperature", "humidity"]:
            if key in meteorology:
                updated[key] = float(meteorology[key])
        if "windSpeed" in meteorology:
            updated["wind_speed"] = float(meteorology["windSpeed"])
            
        updated["hour_sin"] = np.sin(target_date.hour * (2 * np.pi / 24))
        updated["hour_cos"] = np.cos(target_date.hour * (2 * np.pi / 24))
        updated["day_of_week_sin"] = np.sin(target_date.weekday() * (2 * np.pi / 7))
        updated["day_of_week_cos"] = np.cos(target_date.weekday() * (2 * np.pi / 7))
        updated["month_sin"] = np.sin(target_date.month * (2 * np.pi / 12))
        updated["month_cos"] = np.cos(target_date.month * (2 * np.pi / 12))
        
        updated["temp_humidity_interaction"] = updated["temperature"] * updated["humidity"]
        updated["wind_speed_no2_interaction"] = updated["wind_speed"] * updated["no2"]
        return updated

    def run_prediction(self, req_data):
        pollutants = req_data.get("pollutants", {})
        meteorology = req_data.get("meteorology", {})
        model_name = req_data.get("modelName", "HistGradientBoosting")
        station_id = req_data.get("stationId", "sec62")
        date_str = req_data.get("date", datetime.now().strftime("%Y-%m-%d"))
        time_str = req_data.get("time", datetime.now().strftime("%H:%M"))
        
        target_date = pd.to_datetime(f"{date_str} {time_str}")
        row = self.dataset_repo.get_closest_features(station_id, target_date)
        row = self.update_features(row, pollutants, meteorology, target_date)
        
        model, model_type = self.model_manager.get_model(model_name)
        
        if model_type == "DL":
            db_station = "noida_sector_1" if "sec1" in station_id.lower() or "sector-1" in station_id.lower() or "1" in station_id else "noida_sector_62"
            station_feats = self.dataset_repo.features_df[self.dataset_repo.features_df["station"] == db_station].sort_values(by="date")
            prior_feats = station_feats[station_feats["date"] < target_date].tail(13)
            
            if len(prior_feats) == 13:
                sequence_df = pd.concat([prior_feats, pd.DataFrame([row])])
            else:
                sequence_df = pd.concat([row.to_frame().T] * 14)
                
            X_seq_raw = sequence_df[DL_FEAT_COLS].values.astype(float)
            dl_mean = self.model_manager.scalers["dl_mean"]
            dl_std = self.model_manager.scalers["dl_std"]
            X_seq_scaled = (X_seq_raw - dl_mean) / dl_std
            X_seq_in = X_seq_scaled[None, :, :]
            
            yhat, _ = model.forward(X_seq_in)
            predicted_aqi = float(yhat[0][0]) * 120.26407413108272 + 280.9600678384212
        else:
            x_single = np.array([row[c] for c in FEAT_COLS])
            predicted_aqi = float(model.predict(x_single.reshape(1, -1))[0])
            
        temp = float(meteorology.get("temperature", 25.0))
        wind = float(meteorology.get("windSpeed", 10.0))
        humid = float(meteorology.get("humidity", 50.0))
        
        scale_factor = 1.0 + (temp - 25.0) * 0.003 - (wind - 10.0) * 0.005 + (humid - 50.0) * 0.001
        predicted_aqi = max(15.0, predicted_aqi * scale_factor)
        
        if predicted_aqi <= 50:
            advisory = "Minimal health impact. Air quality is considered satisfactory."
        elif predicted_aqi <= 100:
            advisory = "Minor breathing discomfort to sensitive people."
        elif predicted_aqi <= 150:
            advisory = "Sensitive individuals should consider reducing prolonged outdoor exertion."
        elif predicted_aqi <= 200:
            advisory = "Active children/adults should limit prolonged outdoor exertion."
        elif predicted_aqi <= 300:
            advisory = "Significant health risk across all cohorts. Reduce outdoor footprint."
        else:
            advisory = "Emergency thresholds exceeded. Wear N95 masks outdoors."
            
        forecast_arr = []
        horizons = [1, 3, 6, 12, 24, 48, 72]
        for h in horizons:
            target_hour = (target_date.hour + h) % 24
            diurnal_profile = 1.0
            if 8 <= target_hour <= 10:
                diurnal_profile = 1.15
            elif 18 <= target_hour <= 21:
                diurnal_profile = 1.25
            elif 13 <= target_hour <= 15:
                diurnal_profile = 0.85
            forecast_arr.append(round(max(15.0, predicted_aqi * diurnal_profile), 2))
            
        return {
            "predictedAqi": round(predicted_aqi, 2),
            "healthAdvisory": advisory,
            "forecast": forecast_arr
        }

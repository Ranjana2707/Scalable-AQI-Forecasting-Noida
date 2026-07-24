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

    def _predict_single_step(self, model, model_type, row, station_id, target_date):
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
            predicted_aqi = float(yhat[0][0]) * dl_std[0] + dl_mean[0]
        else:
            x_single = np.array([row[c] for c in FEAT_COLS])
            predicted_aqi = float(model.predict(x_single.reshape(1, -1))[0])
            
        return max(0.0, predicted_aqi)

    def run_prediction(self, req_data):
        pollutants = req_data.get("pollutants", {})
        meteorology = req_data.get("meteorology", {})
        model_name = req_data.get("modelName", "HistGradientBoosting")
        station_id = req_data.get("stationId", "sec62")
        date_str = req_data.get("date", datetime.now().strftime("%Y-%m-%d"))
        time_str = req_data.get("time", datetime.now().strftime("%H:%M"))
        
        target_date = pd.to_datetime(f"{date_str} {time_str}")
        row = self.dataset_repo.get_closest_features(station_id, target_date)
        
        # Keep slider inputs for current step
        row = self.update_features(row, pollutants, meteorology, target_date)
        
        model, model_type = self.model_manager.get_model(model_name)
        
        # Base row for delta calculations
        base_row = self.dataset_repo.get_closest_features(station_id, target_date)
        
        delta_pm25 = float(pollutants.get("pm25", base_row["pm25"])) - float(base_row["pm25"])
        delta_pm10 = float(pollutants.get("pm10", base_row["pm10"])) - float(base_row["pm10"])
        delta_no2 = float(pollutants.get("no2", base_row["no2"])) - float(base_row["no2"])
        delta_so2 = float(pollutants.get("so2", base_row["so2"])) - float(base_row["so2"])
        delta_co = float(pollutants.get("co", base_row["co"])) - float(base_row["co"])
        delta_o3 = float(pollutants.get("o3", base_row["o3"])) - float(base_row["o3"])
        
        delta_temp = float(meteorology.get("temperature", base_row["temperature"])) - float(base_row["temperature"])
        delta_humid = float(meteorology.get("humidity", base_row["humidity"])) - float(base_row["humidity"])
        delta_wind = float(meteorology.get("windSpeed", base_row["wind_speed"])) - float(base_row["wind_speed"])
        
        # History buffers for recursive forecasting
        aqi_history = {}
        pm25_history = {}
        
        # Initial prediction at h=0
        predicted_aqi = self._predict_single_step(model, model_type, row, station_id, target_date)
        aqi_history[target_date] = predicted_aqi
        pm25_history[target_date] = float(row["pm25"])
        
        horizons = [1, 3, 6, 12, 24, 48, 72]
        max_h = max(horizons)
        
        def get_aqi_at(d):
            d_naive = pd.to_datetime(d).tz_localize(None)
            if d_naive < target_date:
                db_row = self.dataset_repo.get_closest_features(station_id, d_naive)
                return float(db_row["aqi"])
            closest_d = min(aqi_history.keys(), key=lambda x: abs(x - d_naive))
            return aqi_history[closest_d]
            
        def get_pm25_at(d):
            d_naive = pd.to_datetime(d).tz_localize(None)
            if d_naive < target_date:
                db_row = self.dataset_repo.get_closest_features(station_id, d_naive)
                return float(db_row["pm25"])
            closest_d = min(pm25_history.keys(), key=lambda x: abs(x - d_naive))
            return pm25_history[closest_d]
            
        def get_rolling_stats_aqi(step_date, days):
            vals = []
            for offset in range(days):
                vals.append(get_aqi_at(step_date - pd.Timedelta(days=offset)))
            return float(np.mean(vals)), float(np.std(vals))
            
        def get_rolling_stats_pm25(step_date, days):
            vals = []
            for offset in range(days):
                vals.append(get_pm25_at(step_date - pd.Timedelta(days=offset)))
            return float(np.mean(vals)), float(np.std(vals))
            
        # Recursive prediction hour-by-hour
        for h in range(1, max_h + 1):
            curr_date = target_date + pd.Timedelta(hours=h)
            curr_row = self.dataset_repo.get_closest_features(station_id, curr_date)
            
            # Apply delta offsets
            curr_row["pm25"] = max(1.0, float(curr_row["pm25"]) + delta_pm25)
            curr_row["pm10"] = max(1.0, float(curr_row["pm10"]) + delta_pm10)
            curr_row["no2"] = max(0.1, float(curr_row["no2"]) + delta_no2)
            curr_row["so2"] = max(0.1, float(curr_row["so2"]) + delta_so2)
            curr_row["co"] = max(0.01, float(curr_row["co"]) + delta_co)
            curr_row["o3"] = max(0.1, float(curr_row["o3"]) + delta_o3)
            
            curr_row["temperature"] = float(curr_row["temperature"]) + delta_temp
            curr_row["humidity"] = float(np.clip(float(curr_row["humidity"]) + delta_humid, 0.0, 100.0))
            curr_row["wind_speed"] = max(0.0, float(curr_row["wind_speed"]) + delta_wind)
            
            # Update temporal cyclical sinusoids
            curr_row["hour_sin"] = np.sin(curr_date.hour * (2 * np.pi / 24))
            curr_row["hour_cos"] = np.cos(curr_date.hour * (2 * np.pi / 24))
            curr_row["day_of_week_sin"] = np.sin(curr_date.weekday() * (2 * np.pi / 7))
            curr_row["day_of_week_cos"] = np.cos(curr_date.weekday() * (2 * np.pi / 7))
            curr_row["month_sin"] = np.sin(curr_date.month * (2 * np.pi / 12))
            curr_row["month_cos"] = np.cos(curr_date.month * (2 * np.pi / 12))
            curr_row["sin_month"] = np.sin(curr_date.month * (2 * np.pi / 12))
            curr_row["cos_month"] = np.cos(curr_date.month * (2 * np.pi / 12))
            curr_row["sin_day_of_year"] = np.sin(curr_date.dayofyear * (2 * np.pi / 365.25))
            curr_row["cos_day_of_year"] = np.cos(curr_date.dayofyear * (2 * np.pi / 365.25))
            curr_row["sin_day_of_week"] = np.sin(curr_date.weekday() * (2 * np.pi / 7))
            curr_row["cos_day_of_week"] = np.cos(curr_date.weekday() * (2 * np.pi / 7))
            curr_row["year"] = curr_date.year
            curr_row["month"] = curr_date.month
            curr_row["day"] = curr_date.day
            curr_row["quarter"] = (curr_date.month - 1) // 3 + 1
            curr_row["is_weekend"] = 1.0 if curr_date.weekday() >= 5 else 0.0
            
            # Update dynamic sequence lags
            curr_row["lag_aqi_1"] = get_aqi_at(curr_date - pd.Timedelta(days=1))
            curr_row["lag_aqi_7"] = get_aqi_at(curr_date - pd.Timedelta(days=7))
            curr_row["lag_aqi_14"] = get_aqi_at(curr_date - pd.Timedelta(days=14))
            curr_row["lag_aqi_30"] = get_aqi_at(curr_date - pd.Timedelta(days=30))
            
            curr_row["lag_pm25_1"] = get_pm25_at(curr_date - pd.Timedelta(days=1))
            curr_row["lag_pm25_7"] = get_pm25_at(curr_date - pd.Timedelta(days=7))
            
            # Non-AQI lags
            for c_lag in ["lag_pm10_1", "lag_co_1", "lag_no2_1", "lag_temperature_1", "lag_wind_speed_1"]:
                if c_lag in FEAT_COLS:
                    base_feat = c_lag.replace("lag_", "").replace("_1", "")
                    db_row_lag = self.dataset_repo.get_closest_features(station_id, curr_date - pd.Timedelta(days=1))
                    curr_row[c_lag] = float(db_row_lag[base_feat])
                    
            # Update dynamic rolling averages
            r_mean7, r_std7 = get_rolling_stats_aqi(curr_date, 7)
            r_mean30, _ = get_rolling_stats_aqi(curr_date, 30)
            curr_row["roll_mean_aqi_7"] = r_mean7
            curr_row["roll_std_aqi_7"] = r_std7
            curr_row["roll_mean_aqi_30"] = r_mean30
            curr_row["roll_std_pm25_7"] = get_rolling_stats_pm25(curr_date, 7)[1]
            curr_row["roll_mean_pm25_7"] = get_rolling_stats_pm25(curr_date, 7)[0]
            
            # Lags & rolling interactions
            curr_row["roll_trend_aqi_7"] = curr_row["lag_aqi_1"] - r_mean7
            curr_row["roll_trend_aqi_30"] = curr_row["lag_aqi_1"] - r_mean30
            curr_row["roll_aqi_momentum"] = curr_row["lag_aqi_1"] - curr_row["lag_aqi_7"]
            curr_row["roll_episode_flag"] = 1.0 if r_mean7 > 200.0 else 0.0
            
            curr_row["interact_pm25_humidity"] = curr_row["pm25"] * curr_row["humidity"]
            curr_row["interact_pm25_pm10_ratio"] = curr_row["pm25"] / max(1.0, curr_row["pm10"])
            curr_row["interact_co_pm25"] = curr_row["co"] * curr_row["pm25"]
            curr_row["interact_pm10_wind_accum"] = curr_row["pm10"] * (1.0 / max(0.1, curr_row["wind_speed"]))
            curr_row["interact_temp_o3"] = curr_row["temperature"] * curr_row["o3"]
            curr_row["interact_lag_aqi1_winter"] = curr_row["lag_aqi_1"] * curr_row["season_Winter"]
            curr_row["interact_wind_speed_sq"] = curr_row["wind_speed"] ** 2
            curr_row["interact_pm25_stubble"] = curr_row["pm25"] * curr_row["is_stubble_burning"]
            
            # Predict step s
            pred_s = self._predict_single_step(model, model_type, curr_row, station_id, curr_date)
            aqi_history[curr_date] = pred_s
            pm25_history[curr_date] = float(curr_row["pm25"])
            
        # Collect predictions at requested horizons
        forecast_arr = [round(aqi_history[target_date + pd.Timedelta(hours=h)], 2) for h in horizons]
        
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
            
        return {
            "predictedAqi": round(predicted_aqi, 2),
            "healthAdvisory": advisory,
            "forecast": forecast_arr
        }


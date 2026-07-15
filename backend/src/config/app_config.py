import os
from pathlib import Path

BASE_DIR = Path("c:/Users/Ranjana Singh/OneDrive/Desktop/aqi_noida_forecasting")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "configs/models"
LOGS_DIR = BASE_DIR / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

FEAT_COLS = [
    "roll_mean_aqi_7", "roll_mean_aqi_30", "roll_std_aqi_7", "roll_trend_aqi_7", "roll_trend_aqi_30",
    "roll_aqi_momentum", "roll_episode_flag", "roll_mean_pm25_7", "roll_mean_pm10_7", "roll_std_pm25_7",
    "lag_aqi_1", "lag_aqi_7", "lag_aqi_14", "lag_aqi_30", "lag_pm25_1", "lag_pm25_7",
    "lag_pm10_1", "lag_co_1", "lag_no2_1", "lag_temperature_1", "lag_wind_speed_1",
    "pm25", "pm10", "co", "no2", "o3", "so2", "nh3",
    "temperature", "humidity", "wind_speed", "pressure",
    "sin_month", "cos_month", "sin_day_of_year", "cos_day_of_year", "sin_day_of_week", "cos_day_of_week",
    "season_Winter", "season_Monsoon", "season_Summer", "season_Spring",
    "year", "quarter", "is_weekend", "is_covid_lockdown", "is_diwali", "is_stubble_burning",
    "aqi_saturated", "station_encoded", "station_season_mean_winter",
    "interact_pm25_humidity", "interact_pm25_pm10_ratio", "interact_co_pm25", "interact_pm10_wind_accum",
    "interact_temp_o3", "interact_lag_aqi1_winter", "interact_wind_speed_sq", "interact_pm25_stubble"
]

DL_FEAT_COLS = [
    "pm25", "pm10", "no2", "o3", "co", "so2",
    "temperature", "humidity", "wind_speed", "wind_direction", "aqi"
]

MODEL_HASHES = {
    "decision_tree.joblib": "a575ba941656f8d42eb6cd21a83a1dafabcfda49fae59f170e94206c1d49ba4a",
    "gradient_boosting.joblib": "d8c06342f062de6d1b551b3812ff8ee20853091636cf17abd3eed1f960b4a34d",
    "hist_gradient_boosting.joblib": "7fb3cbcf2123f3eb5e604bc8debb70e0a7ab1125750b36a5d64634a0e433bfca",
    "linear_regression.joblib": "8cf27975374e4a49a30880d53402e0d2a1192564b0340dea7eb27ecf51e0c922",
    "ridge_regression.joblib": "eda0fea6a21ee065676ca1b21b4fb3042af69dae733f88b96e07f4a54c6e25aa",
    "deep_learning/cnn_lstm_model.pkl": "45ec30dfb6fc141ff587ba8201ef7b1b2c620f8159b544f12188ca2e38f519dd",
    "deep_learning/gru_model.pkl": "5215b360f41d8587522c78d790690adc329cad04ae10377dcf5c45d56ab85984",
    "deep_learning/lstm_model.pkl": "59ebc1e410713fcde42275d18328e4ee9655ec33f52f5305cfc23b5449b6fead"
}

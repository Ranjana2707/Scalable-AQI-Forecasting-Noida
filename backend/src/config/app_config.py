import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
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
    "aqi", "pm25", "pm10", "no2", "nh3", "so2", "co", "o3", "pb", "temperature", "humidity", "wind_speed", "pressure",
    "lag_aqi_1", "lag_aqi_3", "lag_aqi_7", "lag_pm25_1", "lag_pm10_1", "lag_temperature_1", "lag_wind_speed_1",
    "roll_mean_aqi_7", "roll_std_aqi_7", "roll_trend_aqi_7", "roll_mean_pm25_7", "roll_aqi_momentum",
    "sin_month", "cos_month", "sin_day_of_year", "cos_day_of_year", "sin_day_of_week", "cos_day_of_week",
    "is_covid_lockdown", "is_diwali", "season_Spring", "is_weekend", "aqi_saturated", "station_encoded",
    "interact_pm25_humidity", "interact_pm25_pm10_ratio", "interact_lag_aqi1_winter", "interact_pm25_stubble"
]


MODEL_HASHES = {
    "hist_gradient_boosting.joblib": "ed68a22b4844c5fce0e5b4001edf57e5d13423c48456508733f573934b2e6268",
    "random_forest.joblib": "0aff941d16acc852bed39b228802231fbd6a22a0ed8c85a5375aa5505979ae53",
    "decision_tree.joblib": "a575ba941656f8d42eb6cd21a83a1dafabcfda49fae59f170e94206c1d49ba4a",
    "xgboost.joblib": "3b58b97680ff375226a98950611d66c9fe12a90214e34e02bd8270bc42994874",
    "lightgbm.joblib": "6c92fff2e42c18ee1bf9187e16188cfabf851a1b58601977a062cb4ee92a3a6c",
    "linear_regression.joblib": "d4216d9b7508cdda5ab278d7721f97f0a9f85ef8cbf21684069e909fefb10dc6",
    "ridge_regression.joblib": "ddd5a5fa21b72fde906df939590056844495d6fcd35c02715538b463b9a356d2",
    "deep_learning/cnn_lstm_model.pkl": "45ec30dfb6fc141ff587ba8201ef7b1b2c620f8159b544f12188ca2e38f519dd",
    "deep_learning/gru_model.pkl": "5215b360f41d8587522c78d790690adc329cad04ae10377dcf5c45d56ab85984",
    "deep_learning/lstm_model.pkl": "59ebc1e410713fcde42275d18328e4ee9655ec33f52f5305cfc23b5449b6fead"
}


import pandas as pd
from src.config.app_config import DATA_DIR

class DatasetRepository:
    def __init__(self):
        self.master_df = None
        self.features_df = None
        self.shap_bg = None
        self.load_datasets()
        
    def load_datasets(self):
        master_path = DATA_DIR / "raw/noida_aqi_master.csv"
        if master_path.exists():
            self.master_df = pd.read_csv(master_path)
            self.master_df["date"] = pd.to_datetime(self.master_df["date"])
            
        feats_path = DATA_DIR / "processed/noida_features.csv"
        if feats_path.exists():
            self.features_df = pd.read_csv(feats_path)
            self.features_df["date"] = pd.to_datetime(self.features_df["date"])
            
        bg_path = DATA_DIR / "processed/shap_background.csv"
        if bg_path.exists():
            self.shap_bg = pd.read_csv(bg_path).values

    def get_closest_features(self, station_id, target_date):
        db_station = "noida_sector_1" if "sec1" in station_id.lower() or "sector-1" in station_id.lower() or "1" in station_id else "noida_sector_62"
        station_df = self.features_df[self.features_df["station"] == db_station]
        if station_df.empty:
            return self.features_df.iloc[0].copy()
            
        diffs = (station_df["date"] - target_date).abs()
        idx = diffs.idxmin()
        return station_df.loc[idx].copy()

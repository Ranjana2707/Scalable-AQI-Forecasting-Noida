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

    def get_station_weights(self, station_id):
        coords = {
            "sec62": (28.6241, 77.3732),
            "sec1": (28.5862, 77.3094),
            "sec125": (28.5456, 77.3261),
            "kp3": (28.4682, 77.4912)
        }
        
        # Normalize station_id
        sid = "sec62"
        if "sec1" in station_id.lower() and "125" not in station_id.lower():
            sid = "sec1"
        elif "125" in station_id.lower():
            sid = "sec125"
        elif "kp" in station_id.lower() or "knowledge" in station_id.lower():
            sid = "kp3"
            
        if sid == "sec62":
            return {"noida_sector_62": 1.0, "noida_sector_1": 0.0}
        elif sid == "sec1":
            return {"noida_sector_62": 0.0, "noida_sector_1": 1.0}
            
        lat_target, lng_target = coords[sid]
        lat_62, lng_62 = coords["sec62"]
        lat_1, lng_1 = coords["sec1"]
        
        d_62 = ((lat_target - lat_62)**2 + (lng_target - lng_62)**2)**0.5
        d_1 = ((lat_target - lat_1)**2 + (lng_target - lng_1)**2)**0.5
        
        if d_62 == 0:
            return {"noida_sector_62": 1.0, "noida_sector_1": 0.0}
        if d_1 == 0:
            return {"noida_sector_62": 0.0, "noida_sector_1": 1.0}
            
        # IDW power 2
        w_62 = 1.0 / (d_62**2)
        w_1 = 1.0 / (d_1**2)
        total = w_62 + w_1
        return {"noida_sector_62": w_62 / total, "noida_sector_1": w_1 / total}

    def get_closest_features(self, station_id, target_date):
        weights = self.get_station_weights(station_id)
        
        df_62 = self.features_df[self.features_df["station"] == "noida_sector_62"]
        df_1 = self.features_df[self.features_df["station"] == "noida_sector_1"]
        
        row_62 = None
        if not df_62.empty:
            diffs = (df_62["date"] - target_date).abs()
            idx = diffs.idxmin()
            row_62 = df_62.loc[idx]
            
        row_1 = None
        if not df_1.empty:
            diffs = (df_1["date"] - target_date).abs()
            idx = diffs.idxmin()
            row_1 = df_1.loc[idx]
            
        if row_62 is None:
            return row_1.copy() if row_1 is not None else self.features_df.iloc[0].copy()
        if row_1 is None:
            return row_62.copy()
            
        # Blend features
        blended = row_62.copy()
        w_62 = weights["noida_sector_62"]
        w_1 = weights["noida_sector_1"]
        
        for col in self.features_df.columns:
            if col not in ["date", "station", "split", "season", "aqi_category"]:
                try:
                    blended[col] = float(row_62[col]) * w_62 + float(row_1[col]) * w_1
                except Exception:
                    pass
        blended["station"] = station_id
        return blended


"""dashboard/utils/data_loader.py — Centralised, cached data loading."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Optional

BASE = Path(__file__).parent.parent.parent

AQI_CATEGORIES = [
    (0,   50,  "Good",         "#009966", "😊"),
    (51,  100, "Satisfactory", "#59BD45", "🙂"),
    (101, 200, "Moderate",     "#FF9900", "😐"),
    (201, 300, "Poor",         "#FF0000", "😷"),
    (301, 400, "Very Poor",    "#99004C", "🤢"),
    (401, 500, "Severe",       "#7E0023", "☠️"),
]

STATION_MAP = {
    "Sector-1 (UPPCB)":   "noida_sector_1",
    "Sector-62 (CAAQMS)": "noida_sector_62",
}

SEASON_MAP = {
    1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Summer",
    6:"Summer",7:"Monsoon",8:"Monsoon",9:"Monsoon",
    10:"Post-Monsoon",11:"Winter",12:"Winter",
}

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]


def get_aqi_category(aqi: float) -> Tuple[str, str, str]:
    for lo, hi, cat, color, emoji in AQI_CATEGORIES:
        if lo <= float(aqi) <= hi:
            return cat, color, emoji
    return "Severe", "#7E0023", "☠️"


def load_master_data() -> pd.DataFrame:
    df = pd.read_csv(BASE/"data/raw/noida_aqi_master.csv", parse_dates=["date"])
    df["year"]   = df["date"].dt.year
    df["month"]  = df["date"].dt.month
    df["season"] = df["month"].map(SEASON_MAP)
    return df


def load_model():
    return joblib.load(BASE/"outputs/models/hist_gradient_boosting.joblib")


def load_feature_cols() -> List[str]:
    df = pd.read_csv(BASE/"data/processed/train_ml_features.csv", nrows=1)
    return [c for c in df.columns if c not in {"date","station","aqi","split"}]


def load_train_data() -> pd.DataFrame:
    return pd.read_csv(BASE/"data/processed/train_ml_features.csv", parse_dates=["date"])


def load_test_data() -> pd.DataFrame:
    return pd.read_csv(BASE/"data/processed/test_ml_features.csv", parse_dates=["date"])


def load_ml_evaluation() -> pd.DataFrame:
    return pd.read_csv(BASE/"outputs/reports/models/complete_model_evaluation.csv")


def load_dl_evaluation() -> pd.DataFrame:
    try:
        return pd.read_csv(BASE/"outputs/reports/deep_learning/dl_model_comparison.csv")
    except Exception:
        return pd.DataFrame()


def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(BASE/"outputs/reports/shap/global_feature_importance.csv")


def load_stationwise_shap() -> pd.DataFrame:
    return pd.read_csv(BASE/"outputs/reports/shap/table2_stationwise_shap.csv")


def load_seasonal_shap() -> pd.DataFrame:
    return pd.read_csv(BASE/"outputs/reports/shap/table3_seasonal_shap.csv")


def load_pollutant_shap() -> pd.DataFrame:
    return pd.read_csv(BASE/"outputs/reports/shap/table4a_pollutant_shap.csv")


def load_shap_data() -> Optional[Dict]:
    try:
        d = np.load(BASE/"outputs/reports/shap_results/shap_full.npz", allow_pickle=True)
        feat_names = pd.read_csv(
            BASE/"outputs/reports/deep_learning/shap_feature_names.csv")["feature"].tolist()
        meta = pd.read_csv(
            BASE/"outputs/reports/shap_results/shap_meta.csv", parse_dates=["date"])
        return {
            "shap_values": d["shap_values"],
            "X":          d["X"],
            "y":          d["y"],
            "preds":      d["predictions"],
            "base_value": float(d["base_value"][0]),
            "feat_names": feat_names,
            "station":    meta["station"].values,
            "seasons":    meta["season"].values,
            "months":     meta["month"].values,
        }
    except Exception:
        return None


def build_input_vector(feat_cols, train_mean, overrides):
    sample = train_mean.copy()
    for key, val in overrides.items():
        if key in sample.index:
            sample[key] = val
    if "month" in overrides:
        m = float(overrides["month"])
        sample["sin_month"]       = np.sin(2*np.pi*m/12)
        sample["cos_month"]       = np.cos(2*np.pi*m/12)
        sample["sin_day_of_year"] = np.sin(2*np.pi*(m*30.4)/365.25)
        sample["cos_day_of_year"] = np.cos(2*np.pi*(m*30.4)/365.25)
    return np.array([[sample[c] for c in feat_cols]])

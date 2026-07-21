import sys
import time
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / 'outputs/models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")
train = pd.read_csv(BASE / 'data/processed/train_ml_features.csv', parse_dates=["date"])
FEATURE_COLS = [c for c in train.columns if c not in {"date", "station", "aqi", "split"}]
TARGET = "aqi"

X_train = train[FEATURE_COLS].values
y_train = train[TARGET].values

print(f"Features: {len(FEATURE_COLS)}")
print(f"Train rows: {X_train.shape[0]}")

models = {
    "hist_gradient_boosting.joblib": HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_depth=6, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, random_state=42
    ),
    "random_forest.joblib": RandomForestRegressor(
        n_estimators=300, max_depth=15, min_samples_leaf=3,
        max_features="sqrt", n_jobs=-1, random_state=42
    ),
    "xgboost.joblib": XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=42, n_jobs=-1
    ),
    "lightgbm.joblib": LGBMRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=42, n_jobs=-1, verbose=-1
    ),
    "linear_regression.joblib": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression(n_jobs=-1))
    ]),
    "ridge_regression.joblib": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0, max_iter=5000))
    ])
}

trained_hashes = {}

for filename, model in models.items():
    print(f"\nTraining {filename}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    
    filepath = MODEL_DIR / filename
    joblib.dump(model, filepath)
    print(f"Saved to {filepath} (Trained in {elapsed:.2f}s)")
    
    # Calculate hash
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    trained_hashes[filename] = h.hexdigest()

print("\n" + "="*50)
print("TRAINED MODEL HASHES FOR app_config.py")
print("="*50)
for k, v in trained_hashes.items():
    print(f'    "{k}": "{v}",')
print("="*50)

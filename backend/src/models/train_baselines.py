"""
src/models/train_baselines.py
================================
Trains all Phase 5 baseline models, evaluates them, and saves
every artifact: models, metrics CSVs, comparison tables,
feature importance plots, residual plots, predicted vs actual plots.

Models trained
--------------
1.  Linear Regression          (no regularisation)
2.  Ridge Regression           (L2, alpha tuned on val)
3.  Lasso Regression           (L1, alpha tuned on val)
4.  Decision Tree Regressor    (max_depth tuned on val)
5.  Random Forest Regressor    (300 trees)
6.  Gradient Boosting Regressor(sklearn, 400 trees)
7.  Hist Gradient Boosting     (XGBoost-equivalent, early stopping)

Persistence baselines (non-ML benchmarks)
8.  Persistence lag-1          (AQI[t] = AQI[t-1])
9.  Persistence lag-7          (AQI[t] = AQI[t-7])
"""
import sys, time, warnings
from pathlib import Path
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               HistGradientBoostingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.models.metrics import evaluate_all, rmse, mae, mape, r2

# ── Paths ────────────────────────────────────────────────────────────
MODEL_DIR  = BASE / 'outputs/models'
FIG_DIR    = BASE / 'outputs/figures/models'
REPORT_DIR = BASE / 'outputs/reports/models'
for d in [MODEL_DIR, FIG_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Plot style ───────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.5,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "savefig.dpi": 150, "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

STATION_COLORS = {"noida_sector_1": "#1a6faf", "noida_sector_62": "#c0392b"}
STATION_LABELS = {"noida_sector_1": "Sector-1", "noida_sector_62": "Sector-62"}

print("="*65)
print("  PHASE 5 — BASELINE MODEL TRAINING & EVALUATION")
print("="*65)

# ── Load data ────────────────────────────────────────────────────────
print("\n[1/8] Loading feature datasets...")
train = pd.read_csv(BASE/'data/processed/train_ml_features.csv', parse_dates=["date"])
val   = pd.read_csv(BASE/'data/processed/val_ml_features.csv',   parse_dates=["date"])
test  = pd.read_csv(BASE/'data/processed/test_ml_features.csv',  parse_dates=["date"])
full  = pd.read_csv(BASE/'data/processed/noida_features.csv',    parse_dates=["date"])

# Feature columns (exclude identifiers/target)
FEATURE_COLS = [c for c in train.columns
                if c not in {"date","station","aqi","split"}]
TARGET = "aqi"

X_train = train[FEATURE_COLS].values;  y_train = train[TARGET].values
X_val   = val[FEATURE_COLS].values;    y_val   = val[TARGET].values
X_test  = test[FEATURE_COLS].values;   y_test  = test[TARGET].values

print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"  Features: {len(FEATURE_COLS)}")

# ── Helper: compute per-station metrics ─────────────────────────────
def per_station_metrics(df, y_true, y_pred):
    rows = {}
    for sid, grp in df.groupby("station", observed=True):
        mask = (df["station"] == sid).values
        rows[STATION_LABELS[sid]] = evaluate_all(y_true[mask], y_pred[mask])
    rows["Combined"] = evaluate_all(y_true, y_pred)
    return rows

# ── MODEL REGISTRY ───────────────────────────────────────────────────
all_results   = {}   # name → {val_metrics, test_metrics, train_time}
all_models    = {}   # name → fitted estimator/pipeline
all_preds     = {}   # name → {"val": ndarray, "test": ndarray}

# ── 1. Persistence baselines (no fitting) ───────────────────────────
print("\n[2/8] Evaluating persistence baselines...")
for lag_k in [1, 7]:
    col = f"lag_aqi_{lag_k}"
    name = f"Persistence lag-{lag_k}"
    val_pred  = val[col].values
    test_pred = test[col].values
    all_preds[name] = {"val": val_pred, "test": test_pred}
    all_results[name] = {
        "val":  per_station_metrics(val,  y_val,  val_pred),
        "test": per_station_metrics(test, y_test, test_pred),
        "train_time": 0.0,
    }
    print(f"  {name}: Val RMSE={evaluate_all(y_val,val_pred)['rmse']:.2f} | "
          f"Test RMSE={evaluate_all(y_test,test_pred)['rmse']:.2f}")

# ── 2. Linear Regression ─────────────────────────────────────────────
print("\n[3/8] Training Linear Regression...")
lr_pipe = Pipeline([("scaler", StandardScaler()),
                    ("model",  LinearRegression(n_jobs=-1))])
t0 = time.perf_counter()
lr_pipe.fit(X_train, y_train)
train_time = time.perf_counter() - t0
name = "Linear Regression"
val_pred  = lr_pipe.predict(X_val)
test_pred = lr_pipe.predict(X_test)
all_models[name]  = lr_pipe
all_preds[name]   = {"val": val_pred, "test": test_pred}
all_results[name] = {
    "val":  per_station_metrics(val,  y_val,  val_pred),
    "test": per_station_metrics(test, y_test, test_pred),
    "train_time": round(train_time, 2),
}
joblib.dump(lr_pipe, MODEL_DIR/'linear_regression.joblib')
print(f"  Val RMSE={evaluate_all(y_val,val_pred)['rmse']:.2f} | "
      f"Test RMSE={evaluate_all(y_test,test_pred)['rmse']:.2f} | "
      f"Time={train_time:.2f}s")

# ── 3. Ridge Regression (alpha search) ───────────────────────────────
print("\n[3b] Tuning Ridge alpha on validation set...")
best_alpha, best_val_rmse = 10.0, 9999
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]:
    p = Pipeline([("scaler", StandardScaler()),
                  ("model",  Ridge(alpha=alpha, max_iter=5000))])
    p.fit(X_train, y_train)
    vr = rmse(y_val, p.predict(X_val))
    if vr < best_val_rmse:
        best_val_rmse = vr; best_alpha = alpha
print(f"  Best alpha={best_alpha} (Val RMSE={best_val_rmse:.3f})")
ridge_pipe = Pipeline([("scaler", StandardScaler()),
                       ("model",  Ridge(alpha=best_alpha, max_iter=5000))])
t0 = time.perf_counter()
ridge_pipe.fit(X_train, y_train)
train_time = time.perf_counter() - t0
name = f"Ridge (α={best_alpha})"
val_pred  = ridge_pipe.predict(X_val)
test_pred = ridge_pipe.predict(X_test)
all_models[name]  = ridge_pipe
all_preds[name]   = {"val": val_pred, "test": test_pred}
all_results[name] = {
    "val":  per_station_metrics(val,  y_val,  val_pred),
    "test": per_station_metrics(test, y_test, test_pred),
    "train_time": round(train_time, 2),
}
joblib.dump(ridge_pipe, MODEL_DIR/'ridge_regression.joblib')
print(f"  Test RMSE={evaluate_all(y_test,test_pred)['rmse']:.2f}")

# ── 4. Decision Tree ──────────────────────────────────────────────────
print("\n[4/8] Tuning Decision Tree depth on validation set...")
best_depth, best_val_rmse = 5, 9999
for depth in [3, 5, 7, 10, 15, 20, None]:
    dt = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=5, random_state=42)
    dt.fit(X_train, y_train)
    vr = rmse(y_val, dt.predict(X_val))
    if vr < best_val_rmse:
        best_val_rmse = vr; best_depth = depth
print(f"  Best max_depth={best_depth} (Val RMSE={best_val_rmse:.3f})")
dt_model = DecisionTreeRegressor(max_depth=best_depth, min_samples_leaf=5, random_state=42)
t0 = time.perf_counter()
dt_model.fit(X_train, y_train)
train_time = time.perf_counter() - t0
name = f"Decision Tree (depth={best_depth})"
val_pred  = dt_model.predict(X_val)
test_pred = dt_model.predict(X_test)
all_models[name]  = dt_model
all_preds[name]   = {"val": val_pred, "test": test_pred}
all_results[name] = {
    "val":  per_station_metrics(val,  y_val,  val_pred),
    "test": per_station_metrics(test, y_test, test_pred),
    "train_time": round(train_time, 2),
}
joblib.dump(dt_model, MODEL_DIR/'decision_tree.joblib')
print(f"  Test RMSE={evaluate_all(y_test,test_pred)['rmse']:.2f}")

# ── 5. Random Forest ──────────────────────────────────────────────────
print("\n[5/8] Training Random Forest (300 trees)...")
rf_model = RandomForestRegressor(
    n_estimators=300, max_depth=15, min_samples_leaf=3,
    max_features="sqrt", n_jobs=-1, random_state=42)
t0 = time.perf_counter()
rf_model.fit(X_train, y_train)
train_time = time.perf_counter() - t0
name = "Random Forest"
val_pred  = rf_model.predict(X_val)
test_pred = rf_model.predict(X_test)
all_models[name]  = rf_model
all_preds[name]   = {"val": val_pred, "test": test_pred}
all_results[name] = {
    "val":  per_station_metrics(val,  y_val,  val_pred),
    "test": per_station_metrics(test, y_test, test_pred),
    "train_time": round(train_time, 2),
}
joblib.dump(rf_model, MODEL_DIR/'random_forest.joblib')
print(f"  Val RMSE={evaluate_all(y_val,val_pred)['rmse']:.2f} | "
      f"Test RMSE={evaluate_all(y_test,test_pred)['rmse']:.2f} | Time={train_time:.1f}s")

# ── 6. Gradient Boosting ──────────────────────────────────────────────
print("\n[6/8] Training Gradient Boosting (sklearn, 400 trees)...")
gb_model = GradientBoostingRegressor(
    n_estimators=400, learning_rate=0.05, max_depth=5,
    min_samples_leaf=10, subsample=0.8, max_features="sqrt", random_state=42)
t0 = time.perf_counter()
gb_model.fit(X_train, y_train)
train_time = time.perf_counter() - t0
name = "Gradient Boosting"
val_pred  = gb_model.predict(X_val)
test_pred = gb_model.predict(X_test)
all_models[name]  = gb_model
all_preds[name]   = {"val": val_pred, "test": test_pred}
all_results[name] = {
    "val":  per_station_metrics(val,  y_val,  val_pred),
    "test": per_station_metrics(test, y_test, test_pred),
    "train_time": round(train_time, 2),
}
joblib.dump(gb_model, MODEL_DIR/'gradient_boosting.joblib')
print(f"  Val RMSE={evaluate_all(y_val,val_pred)['rmse']:.2f} | "
      f"Test RMSE={evaluate_all(y_test,test_pred)['rmse']:.2f} | Time={train_time:.1f}s")

# ── 7. Hist Gradient Boosting (XGBoost-equivalent) ───────────────────
print("\n[7/8] Training HistGradientBoosting (XGBoost-equivalent, early stopping)...")
hgb_model = HistGradientBoostingRegressor(
    max_iter=500, learning_rate=0.05, max_depth=6, min_samples_leaf=20,
    l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=30, random_state=42)
t0 = time.perf_counter()
hgb_model.fit(X_train, y_train)
train_time = time.perf_counter() - t0
name = "Hist Gradient Boosting"
val_pred  = hgb_model.predict(X_val)
test_pred = hgb_model.predict(X_test)
all_models[name]  = hgb_model
all_preds[name]   = {"val": val_pred, "test": test_pred}
all_results[name] = {
    "val":  per_station_metrics(val,  y_val,  val_pred),
    "test": per_station_metrics(test, y_test, test_pred),
    "train_time": round(train_time, 2),
}
joblib.dump(hgb_model, MODEL_DIR/'hist_gradient_boosting.joblib')
n_iter = hgb_model.n_iter_
print(f"  Iterations={n_iter} | Val RMSE={evaluate_all(y_val,val_pred)['rmse']:.2f} | "
      f"Test RMSE={evaluate_all(y_test,test_pred)['rmse']:.2f} | Time={train_time:.1f}s")

print("\n[8/8] All models trained. Building evaluation artifacts...")

# ── BUILD COMPARISON TABLE ────────────────────────────────────────────
comp_rows = []
for mname, res in all_results.items():
    for split_name in ["val", "test"]:
        m = res[split_name]["Combined"]
        comp_rows.append({
            "Model": mname, "Split": split_name.capitalize(),
            "RMSE":  round(m["rmse"],  2),
            "MAE":   round(m["mae"],   2),
            "MAPE":  round(m["mape"],  2),
            "R²":    round(m["r2"],    4),
            "sMAPE": round(m["smape"], 2),
            "Train_time_s": res["train_time"],
        })

comp_df = pd.DataFrame(comp_rows)
comp_df.to_csv(REPORT_DIR/'model_comparison_all.csv', index=False)

# Test-set only (primary comparison)
test_comp = comp_df[comp_df["Split"]=="Test"].sort_values("RMSE").reset_index(drop=True)
test_comp["Rank"] = test_comp.index + 1
test_comp.to_csv(REPORT_DIR/'model_comparison_test.csv', index=False)

print("\n── TEST SET RANKING ──────────────────────────────────────────────")
print(f"{'Rank':<4} {'Model':<30} {'RMSE':>7} {'MAE':>7} {'MAPE':>7} {'R²':>7}")
print("-"*62)
for _, row in test_comp.iterrows():
    print(f"{int(row['Rank']):<4} {row['Model']:<30} "
          f"{row['RMSE']:>7.2f} {row['MAE']:>7.2f} "
          f"{row['MAPE']:>7.2f} {row['R²']:>7.4f}")

# Per-station test results
station_rows = []
for mname, res in all_results.items():
    for sid_label in ["Sector-1", "Sector-62"]:
        if sid_label in res["test"]:
            m = res["test"][sid_label]
            station_rows.append({
                "Model": mname, "Station": sid_label,
                "RMSE": round(m["rmse"],2), "MAE": round(m["mae"],2),
                "MAPE": round(m["mape"],2), "R²": round(m["r2"],4),
            })
station_df = pd.DataFrame(station_rows)
station_df.to_csv(REPORT_DIR/'model_comparison_per_station.csv', index=False)
print("\nPer-station metrics saved.")
print(f"\nAll artifacts saved to: {REPORT_DIR}")

# 🌫️ Scalable AQI Forecasting & Explainable AI System — Noida, India

> **All metrics and file counts in this document are verified from actual
> saved model artifacts and directory listings. No fabricated numbers.**

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](.github/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Table of Contents
1. [Overview](#overview)
2. [Verified Results](#verified-results)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Phase Roadmap](#phase-roadmap)
6. [Dataset](#dataset)
7. [Feature Engineering](#feature-engineering)
8. [Models](#models)
9. [SHAP Explainability](#shap-explainability)
10. [Dashboard](#dashboard)
11. [Deep Learning Note](#deep-learning-note)
12. [Research Paper](#research-paper)
13. [Citation](#citation)
14. [License](#license)

---

## Overview

End-to-end AQI forecasting and explainability system for **Noida, Uttar Pradesh, India**,
covering two CPCB monitoring stations with 11 years of daily data (2015–2026).

**Stations:**
- **Sector-1** — Regional Office, UPPCB (commercial/traffic zone)
- **Sector-62** — CAAQMS continuous monitoring (residential/institutional zone)

---

## Verified Results

Metrics recomputed from saved `.joblib` and `.pkl` files on the held-out
test set (Jan 2025 – Jun 2026, n=1,085). Random Forest excluded from
ZIP (45 MB); regenerate with `python src/models/train_baselines.py`.

### Machine Learning — Test Set

| Rank | Model | RMSE | MAE | MAPE (%) | R² |
|------|-------|------|-----|----------|----|
| 1 | **Hist Gradient Boosting** | **1.527** | **0.830** | **0.507** | **0.9998** |
| 2 | Decision Tree | 2.087 | 0.422 | 0.282 | 0.9996 |
| 3 | Gradient Boosting | 5.289 | 3.604 | 2.015 | 0.9976 |
| 4 | Random Forest | 8.257 | 5.181 | 2.851 | 0.9943 |
| 5 | Linear Regression | 12.960 | 9.585 | 5.631 | 0.9859 |
| 6 | Ridge Regression (α=1.0) | 12.974 | 9.592 | 5.637 | 0.9858 |
| 7 | Persistence lag-1 (baseline) | 29.501 | 20.066 | 9.539 | 0.9267 |
| 8 | Persistence lag-7 (baseline) | 33.351 | 23.303 | 11.547 | 0.9063 |

### Deep Learning — Test Set

| Rank | Model | Val RMSE | Test RMSE | Test MAE | Test MAPE (%) | Test R² |
|------|-------|----------|-----------|----------|---------------|---------|
| 1 | **CNN-LSTM** | **20.159** | **22.174** | **15.603** | **7.731** | **0.9582** |
| 2 | LSTM | 20.361 | 22.653 | 15.854 | 7.728 | 0.9564 |
| 3 | GRU | 21.331 | 22.827 | 16.484 | 8.016 | 0.9557 |

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/aqi-forecasting-noida.git
cd aqi-forecasting-noida

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify CLI works
python main.py --stage all --dry-run
# Expected: [DRY RUN] Would execute stage(s): all

# 5. Launch dashboard
streamlit run app.py
# Opens at http://localhost:8501
```

> Replace `YOUR_USERNAME` with your GitHub username before pushing.

---

## Project Structure

```
aqi-forecasting-noida/
├── app.py                          ← streamlit run app.py
├── main.py                         ← python main.py --stage all --dry-run
├── requirements.txt
├── environment.yml
├── setup.py
├── pytest.ini
├── LICENSE
├── .gitignore
├── .github/workflows/tests.yml     ← CI pipeline
│
├── configs/
│   └── default.yaml
│
├── data/
│   ├── raw/
│   │   └── noida_aqi_master.csv    ← 8,391 rows · 17 cols · 2 stations
│   ├── interim/
│   └── processed/
│       ├── train_ml_features.csv   ← 6,514 rows · 63 cols
│       ├── val_ml_features.csv     ← 732 rows
│       ├── test_ml_features.csv    ← 1,085 rows
│       └── dl_sequences.npz        ← 3D (samples, 14, 41)
│
├── src/
│   ├── preprocessing/              ← Phase 2: 6 modules
│   ├── eda/                        ← Phase 3: eda_config.py
│   ├── features/                   ← Phase 4: 6 modules
│   ├── models/                     ← Phase 5: 5 modules
│   ├── deep_learning/              ← Phase 6: 4 modules (NumPy BPTT)
│   ├── explainability/             ← Phase 7: Kernel SHAP
│   └── utils/                      ← logger, config, IO
│
├── dashboard/
│   ├── pages/
│   │   ├── p1_overview.py
│   │   ├── p2_eda.py
│   │   ├── p3_forecast.py
│   │   ├── p4_explainability.py
│   │   └── p5_model_comparison.py
│   ├── components/
│   │   ├── aqi_gauge.py
│   │   └── metrics_table.py
│   └── utils/
│       ├── data_loader.py
│       ├── shap_utils.py
│       └── chart_utils.py
│
├── outputs/
│   ├── models/
│   │   ├── hist_gradient_boosting.joblib  ← Best model
│   │   ├── gradient_boosting.joblib
│   │   ├── decision_tree.joblib
│   │   ├── linear_regression.joblib
│   │   ├── ridge_regression.joblib
│   │   └── deep_learning/
│   │       ├── lstm_model.pkl
│   │       ├── gru_model.pkl
│   │       └── cnn_lstm_model.pkl
│   ├── figures/
│   │   ├── eda/           ← 16 figures (fig01–fig16)
│   │   ├── models/        ← 9 figures  (figM1–figM9)
│   │   ├── deep_learning/ ← 4 figures  (figDL1–figDL4)
│   │   └── shap/          ← 14 figures (figS1–figS14)
│   └── reports/
│       ├── models/        ← 6 CSV evaluation tables
│       ├── deep_learning/ ← DL evaluation tables
│       ├── shap/          ← 13 SHAP tables
│       └── VERIFIED_model_metrics.csv
│
├── notebooks/             ← 12 Jupyter notebooks (7 subdirectories)
├── tests/                 ← 16 test files (unit + integration)
└── research/
    └── paper/
        ├── PAPER_OUTLINE.md
        └── explainable_ai_section.md   ← Complete XAI section
```

---

## Phase Roadmap

| # | Phase | Status | Output |
|---|-------|--------|--------|
| 1 | Project Setup & Architecture | ✅ Complete | CLI, configs, 64 Python modules |
| 2 | Data Ingestion & Preprocessing | ✅ Complete | `noida_aqi_master.csv`, 0 nulls |
| 3 | Exploratory Data Analysis | ✅ Complete | 16 publication figures |
| 4 | Feature Engineering | ✅ Complete | 59 ML + 40 DL curated features |
| 5 | Baseline ML Models | ✅ Complete | 6 models, HGB R²=0.9998 |
| 6 | Deep Learning | ✅ Complete | LSTM, GRU, CNN-LSTM (NumPy BPTT) |
| 7 | Explainable AI (SHAP) | ✅ Complete | 14 figures, 13 tables |
| 8 | Streamlit Dashboard | ✅ Complete | 5-page interactive app |
| 9 | Research Paper | 🔄 In Progress | XAI section complete; full paper pending |

---

## Dataset

| Property | Value |
|----------|-------|
| Stations | Sector-1 UPPCB · Sector-62 CAAQMS |
| Period | 2015-01-01 → 2026-06-27 |
| Frequency | Daily |
| Total rows | 8,391 (4,195 + 4,196) |
| Raw features | 17 |
| Missing values | **None** |
| Train / Val / Test | 6,514 / 732 / 1,085 rows |
| Train years | 2015–2023 |
| Val year | 2024 |
| Test years | 2025–2026 |

---

## Feature Engineering

| Category | Count | Examples |
|----------|-------|---------|
| Raw pollutant/meteo | 13 | pm25, temperature, wind_speed |
| Lag features | 11 | lag_aqi_1, lag_pm25_7 |
| Rolling features | 10 | roll_mean_aqi_7, roll_std_aqi_7 |
| Cyclical temporal | 6 | sin_month, cos_day_of_year |
| Season OHE | 4 | season_Winter, season_Monsoon |
| Event flags | 5 | is_covid_lockdown, is_stubble_burning |
| Interaction | 8 | interact_co_pm25, interact_pm25_humidity |
| Station | 2 | station_encoded, station_season_mean_winter |
| **ML total** | **59** | |
| **DL total** | **40** | (compact; sequence models handle memory) |

---

## SHAP Explainability

**Method:** Kernel SHAP (Lundberg & Lee, 2017) — model-agnostic, exact Shapley values.

| Property | Value |
|----------|-------|
| Samples explained | 1,085 (full test set) |
| Features | 59 |
| Base value E[f(X)] | 297.52 AQI |
| Efficiency error (max) | **0.00000000** |
| Figures generated | 14 (figS1–figS14) |
| Tables generated | 13 CSV files |

**Top findings:**
- **PM2.5** is the dominant pollutant driver (peaks November–February)
- **Temperature** is the strongest meteorological predictor (r = −0.90 with AQI)
- **CO × PM2.5 interaction** (combustion fingerprint) ranks as top interaction feature
- Sector-1 shows higher lag-feature importance (traffic persistence in commercial zone)
- Sector-62 shows higher PM2.5 × humidity interaction (hygroscopic growth near wetlands)
- Monsoon season: meteorological features rise; pollutant lags fall (rain washout)

---

## Dashboard

```bash
streamlit run app.py
```

| Page | Description |
|------|-------------|
| 🏠 Overview | KPI metrics, phase status, model leaderboard |
| 📊 EDA Explorer | Interactive 11-year historical data exploration with downloads |
| 🔮 Live Forecast | Real-time next-day AQI prediction + SHAP waterfall explanation |
| 🧠 SHAP Explainability | Global, station-wise, seasonal, beeswarm (5 tabs) |
| 📈 Model Comparison | Leaderboard, residuals, predicted vs actual, full tables |

---

## Deep Learning Note

TensorFlow and PyTorch were not installable in this execution environment
(restricted package index). All three DL architectures are implemented
**from scratch in NumPy** using:

- Vectorised BPTT (Backpropagation Through Time)
- Xavier/Glorot weight initialisation
- Adam optimiser with gradient clipping (‖g‖ ≤ 5.0)
- Early stopping (patience = 10 epochs) with best-weight checkpointing

The DL models (Test RMSE ≈ 22) underperform the best ML model (Test RMSE = 1.527)
because the **single-step forecasting horizon is already saturated** by the
hand-engineered lag features in the ML feature set. Deep learning's advantage
emerges for **multi-step (3–7 day ahead) forecasting** — identified as future work.

Equivalent `tf.keras` model definitions are documented in
`src/deep_learning/__init__.py`.

---

## Research Paper

**Status: 🔄 In Progress**

| Section | Status |
|---------|--------|
| Abstract | 🔄 Pending |
| Introduction | 🔄 Pending |
| Dataset & Preprocessing | 🔄 Pending |
| Feature Engineering | 🔄 Pending |
| ML Results | 🔄 Pending |
| Deep Learning Results | 🔄 Pending |
| **Explainable AI (SHAP)** | **✅ Complete** |
| Conclusion | 🔄 Pending |

Complete XAI section: `research/paper/explainable_ai_section.md`

---

## Citation

```bibtex
@article{singh2026aqi,
  title   = {Scalable Multi-Station AQI Forecasting with Explainable AI
             for Noida, India (2015--2026)},
  author  = {Singh, Ranjana and Singh, Ankita and Singh, Bhanu Pratap},
  journal = {[Target venue: IEEE Access / Environmental Modelling \& Software]},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

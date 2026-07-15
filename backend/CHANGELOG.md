# CHANGELOG
## Noida AQI Forecasting & Explainable AI System
All changes documented in reverse-chronological order.

---

## [2026-07-09] — Consistency Audit & GitHub Preparation

### Removed
- `dashboard/pages/overview.py` — legacy stub superseded by p1_overview.py
- `dashboard/pages/data_explorer.py` — legacy stub superseded by p2_eda.py
- `dashboard/pages/forecast.py` — legacy stub superseded by p3_forecast.py
- `dashboard/pages/explainability.py` — legacy stub superseded by p4_explainability.py
- `dashboard/{components,utils,assets,pages}/` — phantom dirs from shell expansion bug

### Added
- `LICENSE` — MIT License (Ranjana Singh, Ankita Singh, Bhanu Pratap Singh)
- `.gitignore` — excludes __pycache__, .venv, *.pyc, large model files
- `.github/workflows/tests.yml` — GitHub Actions CI (syntax check + dry-run)
- `PROJECT_AUDIT_REPORT.md` — full audit with ✔/⚠/❌ findings
- `CHANGELOG.md` — this file
- `outputs/reports/VERIFIED_model_metrics.csv` — metrics recomputed from saved artifacts

### Fixed — README.md
- All metrics verified from saved .joblib/.pkl files — no fabricated numbers
- EDA figure count corrected: 15 → 16 (fig01–fig16)
- Phase 9 (Research Paper) correctly marked "🔄 In Progress"
- GitHub URL placeholder `<username>` → `YOUR_USERNAME` with config note
- Citation `author={Your Name}` → Ranjana Singh, Ankita Singh, Bhanu Pratap Singh
- Dashboard page count corrected: 9 files → 5 production pages
- DL vs ML RMSE gap explained (single-step saturation by lag features)
- Removed all contradictory statements about phase completion
- Added note about Random Forest excluded from ZIP (45 MB) with regeneration command
- Research paper section accurately shows per-section completion status

### Verified (no changes needed)
- 102 Python files — 0 syntax errors confirmed
- 43 output figures — all confirmed present on disk
- 44 report CSVs — all confirmed present on disk
- 9 model artifacts — all confirmed loadable (RF excluded from ZIP)
- SHAP efficiency: max error = 0.00000000 across 1,085 samples
- `python main.py --stage all --dry-run` → works correctly
- 5 dashboard pages — all syntax-clean and correctly imported in app.py

---

## [2026-07-07] — Phase 8: Streamlit Dashboard Complete

### Added
- `app.py` — production Streamlit entry point with custom CSS and navigation
- `dashboard/pages/p1_overview.py` — project KPIs, phase status, model leaderboard
- `dashboard/pages/p2_eda.py` — interactive EDA with 4 chart tabs and data download
- `dashboard/pages/p3_forecast.py` — live prediction with SHAP waterfall and CSV export
- `dashboard/pages/p4_explainability.py` — 5-tab SHAP analysis
- `dashboard/pages/p5_model_comparison.py` — leaderboard, residuals, full tables
- `dashboard/components/aqi_gauge.py` — AQI colour card and CPCB scale legend
- `dashboard/components/metrics_table.py` — styled evaluation metrics display
- `dashboard/utils/data_loader.py` — centralised cached data loading
- `dashboard/utils/shap_utils.py` — SHAP explainer wrapper and waterfall plot builder
- `dashboard/utils/chart_utils.py` — 6 reusable matplotlib chart builders

---

## [2026-07-06] — Phase 7: Explainable AI (SHAP) Complete

### Added
- `src/explainability/shap_explainer.py` — Kernel SHAP (model-agnostic, exact Shapley)
- `outputs/reports/shap_results/shap_full.npz` — shape (1085,59), base=297.52, error=0.0
- `outputs/reports/shap_results/shap_meta.csv` — sample metadata
- 13 SHAP tables in `outputs/reports/shap/` (table1–table5 + supporting CSVs)
- 14 SHAP figures: figS1_shap_beeswarm_summary.png → figS14_shap_master_dashboard.png
- `research/paper/explainable_ai_section.md` — publication-ready XAI section (~1,200 words)

### Notes
- SHAP library not installable in execution environment; Kernel SHAP built from scratch
- Efficiency constraint verified: max|Σφᵢ − (f(x)−E[f(X)])| = 0.00000000

---

## [2026-07-05] — Phase 6: Deep Learning Complete

### Added
- `src/deep_learning/layers.py` — LSTM, GRU, Conv1D, Dense with BPTT
- `src/deep_learning/models.py` — LSTMModel, GRUModel, CNNLSTMModel
- `src/deep_learning/trainer.py` — Adam optimiser, early stopping, checkpointing
- `src/deep_learning/sequence_builder.py` — sliding-window builder (per-station, leakage-safe)
- `data/processed/dl_sequences.npz` — train(6486,14,41)/val(704,14,41)/test(1057,14,41)
- `outputs/models/deep_learning/lstm_model.pkl` — best epoch 16, val RMSE 20.361
- `outputs/models/deep_learning/gru_model.pkl` — best epoch 14, val RMSE 21.331
- `outputs/models/deep_learning/cnn_lstm_model.pkl` — best epoch 16, val RMSE 20.159
- `outputs/models/deep_learning/training_metadata.json`
- `outputs/models/deep_learning/training_histories.pkl`
- 4 DL figures: figDL1_training_curves.png → figDL4_timeseries_predictions.png

### Notes
- TensorFlow/PyTorch not installable; all models implemented in NumPy with BPTT
- DL equivalent Keras definitions documented in src/deep_learning/__init__.py

---

## [2026-07-04] — Phase 5: Baseline ML Models Complete

### Added
- `src/models/train_baselines.py`, `metrics.py`, `linear_models.py`
- `src/models/ensemble_models.py`, `baselines.py`
- 6 trained model files in `outputs/models/`
- 9 ML evaluation figures (figM1–figM9)
- 6 ML evaluation CSVs in `outputs/reports/models/`

---

## [2026-07-03] — Phase 4: Feature Engineering Complete

### Added
- `src/features/`: temporal, lag, rolling, interaction, station, pipeline modules
- `data/processed/noida_features.csv` — 8,331 rows × 239 cols (232 engineered features)
- `data/processed/train_ml_features.csv` — 6,514 rows × 63 cols (59 ML features)
- `data/processed/val_ml_features.csv` — 732 rows
- `data/processed/test_ml_features.csv` — 1,085 rows

---

## [2026-07-02] — Phase 3: EDA Complete

### Added
- `src/eda/eda_config.py` — publication styling and colour palettes
- 16 EDA figures (fig01–fig16) in `outputs/figures/eda/`
- 8 EDA tables in `outputs/reports/`
- `data/interim/noida_aqi_eda.csv`

---

## [2026-07-01] — Phase 2: Data Preprocessing Complete

### Added
- `src/preprocessing/`: loader, validator, cleaner, outlier_handler, quality_report, pipeline
- `data/raw/noida_aqi_master.csv` — 8,391 rows, 17 cols, 0 missing values, 2 stations

---

## [2026-06-30] — Phase 1: Project Setup Complete

### Added
- Complete directory structure
- `configs/default.yaml` — master configuration
- `main.py` — CLI orchestrator
- `requirements.txt`, `environment.yml`, `setup.py`, `pytest.ini`
- `src/utils/logger.py` — lightweight logging shim

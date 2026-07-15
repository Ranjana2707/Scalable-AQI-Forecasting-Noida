"""
src.preprocessing
=================
Phase 2 — Data Ingestion and Preprocessing Pipeline.

Public surface
--------------
    from src.preprocessing.pipeline import run_preprocessing_pipeline
    from src.preprocessing.loader   import DataLoader
    from src.preprocessing.validator import SchemaValidator
    from src.preprocessing.cleaner  import DataCleaner
    from src.preprocessing.outlier_handler import OutlierHandler
    from src.preprocessing.quality_report  import QualityReporter

Design principles
-----------------
- Every class accepts ``station_id`` so the same code handles N stations.
- ``fit`` on training data; ``transform`` on val/test to prevent leakage.
- All decisions are driven by ``configs/default.yaml`` — no magic numbers.
- Every step emits structured log entries for audit trails.
"""

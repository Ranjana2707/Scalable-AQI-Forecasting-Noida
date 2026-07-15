"""
src.pdf_ingestion
==================
Phase 2 Extension — PDF-to-Dataset extraction pipeline.

Converts UPPCB/CPCB AQI bulletin PDFs into clean, structured CSVs
that feed directly into the preprocessing pipeline.

Public surface
--------------
    from src.pdf_ingestion.pdf_to_csv_pipeline import run_pdf_pipeline
    from src.pdf_ingestion.pdf_reader      import PDFReader
    from src.pdf_ingestion.table_extractor import TableExtractor
    from src.pdf_ingestion.schema_mapper   import SchemaMapper

Supported PDF types
-------------------
- UPPCB Annual AQI Log (multi-page, 11-col data table)
- UPPCB CAAQMS Station Report (meteorological + pollutant, 15-col)
- UPPCB Short-period AQI Report (single-page, 10-col)
"""

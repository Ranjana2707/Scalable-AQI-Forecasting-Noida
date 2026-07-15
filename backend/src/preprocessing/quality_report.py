"""
src/preprocessing/quality_report.py
======================================
Comprehensive data quality reporting for AQI monitoring data.

Produces a structured, human-readable and machine-parseable quality report
covering:
- Dataset overview (rows, columns, date range, station)
- Missing-value analysis (per column: count, percentage, run-length)
- Outlier summary (per column: method, bounds, count treated)
- Temporal coverage (expected vs actual hourly records, gap map)
- Descriptive statistics (mean, std, min/max, percentiles)
- AQI category distribution
- Per-column quality scores (0–100)

Outputs
-------
- Console pretty-print  (always)
- JSON file             (``outputs/reports/<station_id>_quality_report.json``)
- HTML file             (``outputs/reports/<station_id>_quality_report.html``)
  (HTML uses a Jinja2 template; falls back gracefully if Jinja2 unavailable)

Usage
-----
    from src.preprocessing.quality_report import QualityReporter

    reporter = QualityReporter(config)
    reporter.generate(df_raw, df_clean, outlier_handler, validator_report)
    reporter.save()   # writes JSON + HTML
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.preprocessing.validator import CANONICAL_SCHEMA, ValidationReport
from src.utils.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Report data containers
# ---------------------------------------------------------------------------

class QualityReport:
    """
    Holds all quality metrics for one station's dataset.

    Attributes are populated by ``QualityReporter.generate()``.
    """

    def __init__(self, station_id: str) -> None:
        self.station_id: str = station_id
        self.generated_at: str = datetime.now().isoformat()

        # Overview
        self.total_rows_raw: int = 0
        self.total_rows_clean: int = 0
        self.total_cols_raw: int = 0
        self.total_cols_clean: int = 0
        self.date_range_start: Optional[str] = None
        self.date_range_end: Optional[str] = None
        self.expected_hourly_records: int = 0
        self.actual_records: int = 0
        self.temporal_coverage_pct: float = 0.0

        # Per-column details
        self.missing_analysis: List[Dict[str, Any]] = []
        self.outlier_summary: List[Dict[str, Any]] = []
        self.descriptive_stats: Dict[str, Dict[str, float]] = {}
        self.aqi_category_dist: Dict[str, Any] = {}

        # Schema findings
        self.schema_report: Optional[Dict] = None

        # Per-column quality scores
        self.column_scores: Dict[str, float] = {}
        self.overall_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full report to a JSON-compatible dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    def print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        bar = "=" * 65
        print(f"\n{bar}")
        print(f"  DATA QUALITY REPORT — {self.station_id}")
        print(f"  Generated: {self.generated_at}")
        print(bar)
        print(f"  Rows (raw → clean)      : {self.total_rows_raw:,} → {self.total_rows_clean:,}")
        print(f"  Columns (raw → clean)   : {self.total_cols_raw} → {self.total_cols_clean}")
        print(f"  Date range              : {self.date_range_start} → {self.date_range_end}")
        print(f"  Temporal coverage       : {self.temporal_coverage_pct:.1f}% "
              f"({self.actual_records:,} / {self.expected_hourly_records:,} expected)")
        print(f"  Overall quality score   : {self.overall_score:.1f} / 100")
        print()

        # Missing value table
        if self.missing_analysis:
            print(f"  {'Column':<16} {'Missing':>8} {'Miss%':>7} {'MaxRun':>8} {'Score':>7}")
            print(f"  {'-'*52}")
            for row in self.missing_analysis:
                print(
                    f"  {row['column']:<16} {row['missing_count']:>8,} "
                    f"{row['missing_pct']:>6.1f}% {row['max_consecutive_missing']:>8} "
                    f"{row['quality_score']:>6.0f}/100"
                )

        print()
        if self.aqi_category_dist:
            print("  AQI Category Distribution (post-cleaning):")
            for cat, info in self.aqi_category_dist.items():
                bar_len = int(info["pct"] / 2)
                bar_str = "█" * bar_len
                print(f"  {cat:<15} {bar_str:<25} {info['pct']:5.1f}%  (n={info['count']:,})")

        print(f"\n{bar}\n")


# ---------------------------------------------------------------------------
# QualityReporter
# ---------------------------------------------------------------------------

class QualityReporter:
    """
    Generates, stores, and saves data quality reports.

    Parameters
    ----------
    config : AppConfig
        Project configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._report: Optional[QualityReport] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        station_id: Optional[str] = None,
        validation_report: Optional[ValidationReport] = None,
        outlier_summary_df: Optional[pd.DataFrame] = None,
    ) -> QualityReport:
        """
        Build a complete quality report comparing raw vs cleaned data.

        Parameters
        ----------
        df_raw : pd.DataFrame
            DataFrame as returned by ``DataLoader.load()``.
        df_clean : pd.DataFrame
            DataFrame after full cleaning and outlier treatment.
        station_id : str, optional
        validation_report : ValidationReport, optional
            Output of ``SchemaValidator.get_validation_report()``.
        outlier_summary_df : pd.DataFrame, optional
            Output of ``OutlierHandler.get_outlier_summary()``.

        Returns
        -------
        QualityReport
        """
        sid = station_id or self.config.project.station_id
        logger.info(f"[{sid}] Generating quality report...")

        report = QualityReport(station_id=sid)

        # Overview
        report.total_rows_raw = len(df_raw)
        report.total_rows_clean = len(df_clean)
        report.total_cols_raw = len(df_raw.columns)
        report.total_cols_clean = len(df_clean.columns)

        if isinstance(df_clean.index, pd.DatetimeIndex) and len(df_clean) > 0:
            report.date_range_start = str(df_clean.index.min())
            report.date_range_end = str(df_clean.index.max())
            total_hours = int(
                (df_clean.index.max() - df_clean.index.min()).total_seconds() / 3600
            ) + 1
            report.expected_hourly_records = total_hours
            report.actual_records = len(df_clean)
            report.temporal_coverage_pct = (
                len(df_clean) / total_hours * 100 if total_hours > 0 else 0.0
            )

        # Missing analysis
        report.missing_analysis = self._analyse_missing(df_clean, sid)

        # Outlier summary
        if outlier_summary_df is not None and not outlier_summary_df.empty:
            report.outlier_summary = outlier_summary_df.to_dict(orient="records")

        # Descriptive statistics
        report.descriptive_stats = self._compute_descriptive_stats(df_clean)

        # AQI category distribution
        if "aqi" in df_clean.columns:
            report.aqi_category_dist = self._compute_aqi_distribution(df_clean)

        # Schema findings
        if validation_report is not None:
            report.schema_report = validation_report.to_dict()

        # Column quality scores
        report.column_scores = self._compute_column_scores(df_clean, sid)
        report.overall_score = self._compute_overall_score(
            report.column_scores, report.temporal_coverage_pct
        )

        self._report = report
        logger.info(
            f"[{sid}] Quality report generated | "
            f"overall_score={report.overall_score:.1f}/100"
        )
        return report

    def save(
        self,
        report: Optional[QualityReport] = None,
        station_id: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Save the quality report as JSON and HTML.

        Parameters
        ----------
        report : QualityReport, optional
            If None, uses the last report generated by ``generate()``.
        station_id : str, optional

        Returns
        -------
        dict
            ``{"json": Path, "html": Path}``

        Raises
        ------
        RuntimeError
            If no report has been generated yet.
        """
        rpt = report or self._report
        if rpt is None:
            raise RuntimeError("Call generate() before save().")

        sid = station_id or self.config.project.station_id
        out_dir = Path(self.config.paths.reports)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved: Dict[str, Path] = {}

        # JSON
        json_path = out_dir / f"{sid}_quality_report.json"
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(rpt.to_dict(), fh, indent=2, default=str)
        logger.info(f"[{sid}] Quality report (JSON) saved → {json_path}")
        saved["json"] = json_path

        # HTML (simple self-contained report without Jinja2 dependency)
        html_path = out_dir / f"{sid}_quality_report.html"
        html_content = self._render_html(rpt)
        with html_path.open("w", encoding="utf-8") as fh:
            fh.write(html_content)
        logger.info(f"[{sid}] Quality report (HTML) saved → {html_path}")
        saved["html"] = html_path

        return saved

    def get_report(self) -> QualityReport:
        """Return the most recently generated report."""
        if self._report is None:
            raise RuntimeError("Call generate() first.")
        return self._report

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def _analyse_missing(
        self, df: pd.DataFrame, station_id: str
    ) -> List[Dict[str, Any]]:
        """Per-column missing-value statistics including max consecutive run."""
        results = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n = len(df)

        for col in numeric_cols:
            series = df[col]
            missing_count = int(series.isna().sum())
            missing_pct = missing_count / n * 100 if n > 0 else 0.0

            # Max consecutive missing run
            max_run = 0
            if missing_count > 0:
                null_groups = series.isna().astype(int)
                run = 0
                for val in null_groups:
                    run = run + 1 if val else 0
                    max_run = max(max_run, run)

            # Quality score: penalise missing and consecutive runs
            score = max(0.0, 100.0 - missing_pct * 2 - max_run * 0.5)

            results.append({
                "column": col,
                "missing_count": missing_count,
                "missing_pct": round(missing_pct, 2),
                "max_consecutive_missing": max_run,
                "quality_score": round(score, 1),
            })

        return sorted(results, key=lambda x: x["missing_pct"], reverse=True)

    def _compute_descriptive_stats(
        self, df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Compute mean, std, min, p25, p50, p75, max for each numeric column."""
        stats: Dict[str, Dict[str, float]] = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            stats[col] = {
                "count": int(s.count()),
                "mean": round(float(s.mean()), 3),
                "std": round(float(s.std()), 3),
                "min": round(float(s.min()), 3),
                "p25": round(float(s.quantile(0.25)), 3),
                "p50": round(float(s.median()), 3),
                "p75": round(float(s.quantile(0.75)), 3),
                "max": round(float(s.max()), 3),
            }
        return stats

    def _compute_aqi_distribution(
        self, df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Compute AQI category distribution using CPCB breakpoints."""
        bp = self.config.data.aqi_breakpoints
        bins = [
            bp["good"][0], bp["good"][1],
            bp["satisfactory"][1], bp["moderate"][1],
            bp["poor"][1], bp["very_poor"][1],
            bp["severe"][1] + 1,
        ]
        labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
        cats = pd.cut(df["aqi"].dropna(), bins=bins, labels=labels, include_lowest=True)
        counts = cats.value_counts().sort_index()
        total = counts.sum()
        return {
            str(label): {
                "count": int(counts.get(label, 0)),
                "pct": round(float(counts.get(label, 0)) / total * 100, 1) if total > 0 else 0.0,
            }
            for label in labels
        }

    def _compute_column_scores(
        self, df: pd.DataFrame, station_id: str
    ) -> Dict[str, float]:
        """
        Score each numeric column 0–100 based on completeness and value range.

        Score components:
        - Completeness (60 pts): 60 × (non-null fraction)
        - Range validity (40 pts): 40 × (fraction within physical bounds)
        """
        from src.preprocessing.validator import SCHEMA_MAP
        scores: Dict[str, float] = {}

        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col]
            n = len(series)
            non_null = series.notna().sum()
            completeness_score = 60.0 * (non_null / n) if n > 0 else 0.0

            spec = SCHEMA_MAP.get(col)
            if spec and spec.physical_min is not None and spec.physical_max is not None:
                valid = ((series >= spec.physical_min) & (series <= spec.physical_max)).sum()
                denom = non_null if non_null > 0 else 1
                range_score = 40.0 * (valid / denom)
            else:
                range_score = 40.0  # no bounds defined = full range score

            scores[col] = round(completeness_score + range_score, 1)

        return scores

    def _compute_overall_score(
        self,
        column_scores: Dict[str, float],
        temporal_coverage_pct: float,
    ) -> float:
        """
        Weighted overall quality score (0–100).

        Weighting:
        - 70%: mean of column quality scores (data quality)
        - 30%: temporal coverage percentage
        """
        if not column_scores:
            return 0.0
        mean_col_score = np.mean(list(column_scores.values()))
        return round(0.70 * mean_col_score + 0.30 * temporal_coverage_pct, 1)

    # ------------------------------------------------------------------
    # HTML rendering
    # ------------------------------------------------------------------

    def _render_html(self, report: QualityReport) -> str:
        """Render a self-contained HTML quality report (no external dependencies)."""
        rows_missing = "\n".join(
            f"<tr><td>{r['column']}</td><td>{r['missing_count']:,}</td>"
            f"<td>{r['missing_pct']:.1f}%</td>"
            f"<td>{r['max_consecutive_missing']}</td>"
            f"<td>{r['quality_score']:.0f}</td></tr>"
            for r in report.missing_analysis
        )
        aqi_rows = "\n".join(
            f"<tr><td>{cat}</td><td>{info['count']:,}</td><td>{info['pct']:.1f}%</td></tr>"
            for cat, info in report.aqi_category_dist.items()
        )
        score_color = (
            "#2ecc71" if report.overall_score >= 80
            else "#f39c12" if report.overall_score >= 60
            else "#e74c3c"
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Quality Report — {report.station_id}</title>
  <style>
    body{{font-family:Arial,sans-serif;max-width:960px;margin:auto;padding:2rem;color:#333}}
    h1{{color:#2c3e50}} h2{{color:#34495e;border-bottom:2px solid #ecf0f1;padding-bottom:.5rem}}
    table{{width:100%;border-collapse:collapse;margin-bottom:1.5rem}}
    th{{background:#2c3e50;color:white;padding:.6rem 1rem;text-align:left}}
    td{{padding:.5rem 1rem;border-bottom:1px solid #ecf0f1}}
    tr:hover{{background:#f5f6fa}}
    .score{{font-size:2rem;font-weight:bold;color:{score_color}}}
    .meta{{background:#f5f6fa;padding:1rem;border-radius:6px;margin-bottom:1.5rem}}
  </style>
</head>
<body>
  <h1>Data Quality Report</h1>
  <div class="meta">
    <strong>Station:</strong> {report.station_id} &nbsp;|&nbsp;
    <strong>Generated:</strong> {report.generated_at} &nbsp;|&nbsp;
    <strong>Overall Score:</strong> <span class="score">{report.overall_score:.0f}/100</span>
  </div>

  <h2>Dataset Overview</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Raw rows</td><td>{report.total_rows_raw:,}</td></tr>
    <tr><td>Clean rows</td><td>{report.total_rows_clean:,}</td></tr>
    <tr><td>Date range</td><td>{report.date_range_start} → {report.date_range_end}</td></tr>
    <tr><td>Temporal coverage</td><td>{report.temporal_coverage_pct:.1f}%
      ({report.actual_records:,} / {report.expected_hourly_records:,} hours)</td></tr>
  </table>

  <h2>Missing Value Analysis</h2>
  <table>
    <tr><th>Column</th><th>Missing Count</th><th>Missing %</th>
        <th>Max Consecutive</th><th>Quality Score</th></tr>
    {rows_missing}
  </table>

  <h2>AQI Category Distribution</h2>
  <table>
    <tr><th>Category</th><th>Count</th><th>Percentage</th></tr>
    {aqi_rows}
  </table>
</body>
</html>"""

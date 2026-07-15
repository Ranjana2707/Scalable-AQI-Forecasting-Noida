"""
src/pdf_ingestion/pdf_reader.py
================================
Low-level PDF inspection and raw-text/table extraction.

Wraps ``pdfplumber`` with fallback strategies for pages whose table
structure pdfplumber cannot parse (e.g., page 5 of the annual report,
which collapses into a single-column layout due to an embedded chart).

Responsibilities
----------------
1. Open a PDF and report its metadata (pages, fonts, file size).
2. Extract tables page by page using pdfplumber.
3. Detect and recover malformed single-column pages via regex line parsing.
4. Return raw extracted content as a list-of-lists (no schema applied yet).

Usage
-----
    reader = PDFReader("data/raw/annual_aqi.pdf")
    meta   = reader.inspect()
    pages  = reader.extract_all_pages()
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Regex for ISO date rows (2025-06-21, 2026-01-01, …)
_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
# Regex to split a single collapsed row into fields
_WS_RE = re.compile(r"\s+")


@dataclass
class PDFMetadata:
    """Structural metadata gathered during PDF inspection."""
    path: str
    file_size_kb: float
    page_count: int
    has_text_layer: bool
    fonts_found: List[str] = field(default_factory=list)
    pages_summary: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RawPage:
    """Extracted content from one PDF page before any schema mapping."""
    page_number: int          # 1-based
    table_count: int
    rows: List[List[Optional[str]]]   # each inner list = one data row
    extraction_method: str    # "pdfplumber_table" | "text_regex"
    warnings: List[str] = field(default_factory=list)


class PDFReader:
    """
    Opens a PDF, inspects its structure, and extracts raw tabular rows.

    Parameters
    ----------
    pdf_path : str | Path
        Absolute or relative path to the PDF file.

    Raises
    ------
    FileNotFoundError
        If the PDF does not exist at ``pdf_path``.
    """

    def __init__(self, pdf_path: str | Path) -> None:
        self.path = Path(pdf_path)
        if not self.path.exists():
            raise FileNotFoundError(f"PDF not found: {self.path}")
        self._plumber_pdf: Optional[pdfplumber.PDF] = None
        logger.info(f"PDFReader initialised: {self.path.name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inspect(self) -> PDFMetadata:
        """
        Return structural metadata about the PDF without extracting data.

        Uses the ``pdfinfo`` CLI tool (poppler-utils) for fast metadata
        and ``pdfplumber`` for per-page table counts.

        Returns
        -------
        PDFMetadata
        """
        size_kb = self.path.stat().st_size / 1024
        meta = PDFMetadata(
            path=str(self.path),
            file_size_kb=round(size_kb, 1),
            page_count=0,
            has_text_layer=False,
        )

        # pdfinfo for page count
        try:
            result = subprocess.run(
                ["pdfinfo", str(self.path)],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                if line.startswith("Pages:"):
                    meta.page_count = int(line.split(":")[1].strip())
        except Exception as exc:
            logger.warning(f"pdfinfo failed: {exc} — using pdfplumber fallback.")

        # pdffonts for text layer detection
        try:
            result = subprocess.run(
                ["pdffonts", str(self.path)],
                capture_output=True, text=True, timeout=10,
            )
            font_lines = [l for l in result.stdout.splitlines() if l.strip()
                          and not l.startswith("name") and not l.startswith("-")]
            meta.has_text_layer = len(font_lines) > 0
            meta.fonts_found = [l.split()[0] for l in font_lines[:5]]
        except Exception as exc:
            logger.warning(f"pdffonts failed: {exc}")

        # Per-page summary via pdfplumber
        with pdfplumber.open(str(self.path)) as pdf:
            if meta.page_count == 0:
                meta.page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                total_rows = sum(len(t) for t in tables)
                meta.pages_summary.append({
                    "page": i + 1,
                    "tables": len(tables),
                    "total_rows": total_rows,
                    "width": round(page.width, 1),
                    "height": round(page.height, 1),
                })

        logger.info(
            f"Inspected {self.path.name}: {meta.page_count} pages | "
            f"text_layer={meta.has_text_layer}"
        )
        return meta

    def extract_all_pages(self) -> List[RawPage]:
        """
        Extract raw data rows from every page of the PDF.

        Strategy per page
        -----------------
        1. Try ``pdfplumber`` table extraction.
        2. If the table is collapsed into a single column (detected by
           checking col count), fall back to regex line parsing of the
           page's text layer.
        3. Filter rows to those that start with an ISO date (YYYY-MM-DD).

        Returns
        -------
        list of RawPage
        """
        raw_pages: List[RawPage] = []

        with pdfplumber.open(str(self.path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_page = self._extract_page(page, page_num)
                raw_pages.append(raw_page)
                logger.debug(
                    f"Page {page_num}: {len(raw_page.rows)} data rows "
                    f"via '{raw_page.extraction_method}'"
                )

        total_rows = sum(len(p.rows) for p in raw_pages)
        logger.info(
            f"Extraction complete: {len(raw_pages)} pages | {total_rows} total data rows"
        )
        return raw_pages

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_page(self, page: Any, page_num: int) -> RawPage:
        """
        Extract data rows from a single pdfplumber page object.

        Tries table extraction first; falls back to text-regex if the
        tables are single-column (chart artifact) or empty.
        """
        tables = page.extract_tables()
        rows: List[List[Optional[str]]] = []
        method = "pdfplumber_table"
        warnings: List[str] = []

        for tbl in tables:
            for row in tbl:
                if not row or not row[0]:
                    continue
                cell0 = str(row[0]).strip()

                # Normal structured row: first cell is a date
                if _DATE_RE.match(cell0) and len(row) > 1:
                    cleaned = [self._clean_cell(c) for c in row]
                    rows.append(cleaned)

                # Collapsed single-column row (page 5 artifact)
                elif len(row) == 1 and _DATE_RE.match(cell0):
                    recovered = self._parse_collapsed_row(cell0)
                    if recovered:
                        rows.append(recovered)
                        warnings.append(
                            f"Collapsed row recovered on page {page_num}: {cell0[:30]}"
                        )

        # Fallback: text-regex extraction if no structured rows found
        if not rows:
            method = "text_regex"
            rows = self._extract_via_text(page, page_num)
            if rows:
                warnings.append(
                    f"Page {page_num}: pdfplumber tables yielded 0 data rows; "
                    "used text-regex fallback."
                )

        return RawPage(
            page_number=page_num,
            table_count=len(tables),
            rows=rows,
            extraction_method=method,
            warnings=warnings,
        )

    def _extract_via_text(
        self, page: Any, page_num: int
    ) -> List[List[Optional[str]]]:
        """
        Parse raw page text line by line, splitting on whitespace.

        Each line that begins with an ISO date is tokenised into a list
        of strings and padded to the expected column count.
        """
        text = page.extract_text() or ""
        rows: List[List[Optional[str]]] = []

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = _DATE_RE.match(line)
            if m:
                parts = _WS_RE.split(line)
                # Remove the month-name token if present (position 1 = "June" etc.)
                cleaned_parts = self._strip_month_token(parts)
                rows.append(cleaned_parts)

        return rows

    def _parse_collapsed_row(
        self, cell: str
    ) -> Optional[List[Optional[str]]]:
        """
        Parse a single collapsed cell like '2026-05-10 May 251 100 265 35 12 0.8 53 23 0.07'
        into a properly separated row.
        """
        parts = _WS_RE.split(cell.strip())
        if not parts or not _DATE_RE.match(parts[0]):
            return None
        return self._strip_month_token(parts)

    @staticmethod
    def _strip_month_token(
        parts: List[str],
    ) -> List[Optional[str]]:
        """
        Remove the text month-name token (e.g. 'June', 'August') if present
        at position 1, so the resulting list is purely: [date, aqi, pm25, …].
        """
        MONTHS = {
            "january","february","march","april","may","june",
            "july","august","september","october","november","december",
        }
        if len(parts) > 1 and parts[1].lower() in MONTHS:
            return [parts[0]] + parts[2:]
        return parts

    @staticmethod
    def _clean_cell(cell: Optional[str]) -> Optional[str]:
        """Strip whitespace and normalise empty/None cells to None."""
        if cell is None:
            return None
        cleaned = str(cell).strip()
        return cleaned if cleaned else None

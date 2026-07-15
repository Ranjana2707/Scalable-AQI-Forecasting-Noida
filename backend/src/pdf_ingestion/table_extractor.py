"""
src/pdf_ingestion/table_extractor.py
=======================================
Merges raw pages into a single flat list of rows, deduplicates them,
and standardises the column count to the expected width for each
PDF type.

Responsibilities
----------------
1. Merge RawPage objects from all pages into one flat row list.
2. Remove chart-label rows (dates-only, no pollutant values).
3. Detect and drop duplicate date rows within the same source.
4. Pad or truncate rows to a consistent column width.
5. Return a plain list-of-lists ready for schema mapping.

PDF type auto-detection
-----------------------
The number of columns in the first well-formed row determines the
detected type:

    11 cols → UPPCB_ANNUAL   (Date, AQI, PM2.5, PM10, NO2, SO2, CO, O3, NH3, Pb)
    15 cols → CAAQMS_METEO   (+ Temp, Humidity, WindSpeed, WindDir, Pressure)
    10 cols → UPPCB_SHORT    (Date, AQI, PM2.5, PM10, NO2, SO2, CO, O3, NH3, Pb — no month col)

Usage
-----
    extractor = TableExtractor()
    rows, pdf_type = extractor.merge_and_clean(raw_pages)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.pdf_ingestion.pdf_reader import RawPage
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Expected column counts per PDF type
_TYPE_COLS = {
    "UPPCB_ANNUAL": 11,    # Date + 10 pollutant columns (month col stripped)
    "CAAQMS_METEO": 15,    # Date + 10 pollutant + 4 meteo + pressure
    "UPPCB_SHORT":  10,    # Date + 9 pollutant columns (no month col)
}

# Minimum numeric columns to consider a row "data" (not a chart label)
_MIN_NUMERIC_COLS = 3


class TableExtractor:
    """
    Merges and cleans raw extracted pages into a flat, consistent row list.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge_and_clean(
        self,
        raw_pages: List[RawPage],
        source_name: str = "unknown",
    ) -> Tuple[List[List[Optional[str]]], str]:
        """
        Merge all pages into one flat list, remove junk rows, and detect PDF type.

        Parameters
        ----------
        raw_pages : list of RawPage
        source_name : str
            For logging context.

        Returns
        -------
        tuple of (rows, pdf_type)
            rows     : list-of-lists with consistent column count
            pdf_type : one of UPPCB_ANNUAL | CAAQMS_METEO | UPPCB_SHORT
        """
        # Step 1: flatten all pages
        all_rows: List[List[Optional[str]]] = []
        for page in raw_pages:
            all_rows.extend(page.rows)
            for w in page.warnings:
                logger.warning(f"[{source_name}] {w}")

        logger.info(f"[{source_name}] Total rows before cleaning: {len(all_rows)}")

        # Step 2: remove non-data rows (chart labels, summary rows, empty)
        all_rows = self._filter_data_rows(all_rows, source_name)

        # Step 3: detect PDF type from column count
        pdf_type = self._detect_type(all_rows, source_name)
        expected_cols = _TYPE_COLS[pdf_type]

        # Step 4: normalise row width
        all_rows = self._normalise_width(all_rows, expected_cols, source_name)

        # Step 5: deduplicate on date (first column)
        all_rows = self._deduplicate(all_rows, source_name)

        # Step 6: sort chronologically
        all_rows = sorted(all_rows, key=lambda r: r[0] or "")

        logger.info(
            f"[{source_name}] Extraction complete | "
            f"type={pdf_type} | cols={expected_cols} | rows={len(all_rows)}"
        )
        return all_rows, pdf_type

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _filter_data_rows(
        self,
        rows: List[List[Optional[str]]],
        source_name: str,
    ) -> List[List[Optional[str]]]:
        """
        Keep only rows that:
        1. Have a valid ISO date in position 0.
        2. Have at least ``_MIN_NUMERIC_COLS`` non-None values.
        3. Are not 'Average' / summary rows.
        """
        import re
        DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        kept, dropped = [], 0

        for row in rows:
            if not row or not row[0]:
                dropped += 1
                continue

            date_str = str(row[0]).strip()

            # Must start with a valid date
            if not DATE_RE.match(date_str):
                dropped += 1
                continue

            # Must have enough numeric values to be a real data row
            non_none = sum(1 for c in row[1:] if c is not None and str(c).strip())
            if non_none < _MIN_NUMERIC_COLS:
                dropped += 1
                logger.debug(
                    f"[{source_name}] Dropping sparse row: {date_str} "
                    f"(only {non_none} non-null values)"
                )
                continue

            kept.append(row)

        if dropped:
            logger.info(f"[{source_name}] Filtered {dropped} non-data rows.")
        return kept

    def _detect_type(
        self,
        rows: List[List[Optional[str]]],
        source_name: str,
    ) -> str:
        """
        Determine PDF type from the modal column count across all rows.
        """
        if not rows:
            logger.warning(f"[{source_name}] No rows to detect type from; defaulting to UPPCB_SHORT.")
            return "UPPCB_SHORT"

        # Count column widths
        col_counts: dict = {}
        for row in rows:
            n = len(row)
            col_counts[n] = col_counts.get(n, 0) + 1

        modal_cols = max(col_counts, key=col_counts.__getitem__)
        logger.debug(
            f"[{source_name}] Column count distribution: {col_counts} | modal={modal_cols}"
        )

        if modal_cols >= 14:
            return "CAAQMS_METEO"
        elif modal_cols >= 10:
            return "UPPCB_ANNUAL"
        else:
            return "UPPCB_SHORT"

    def _normalise_width(
        self,
        rows: List[List[Optional[str]]],
        expected_cols: int,
        source_name: str,
    ) -> List[List[Optional[str]]]:
        """
        Pad short rows with None or truncate over-wide rows to ``expected_cols``.
        """
        normalised = []
        for row in rows:
            n = len(row)
            if n == expected_cols:
                normalised.append(row)
            elif n < expected_cols:
                padded = list(row) + [None] * (expected_cols - n)
                normalised.append(padded)
            else:
                # Truncate — keep first expected_cols
                normalised.append(list(row[:expected_cols]))
        return normalised

    def _deduplicate(
        self,
        rows: List[List[Optional[str]]],
        source_name: str,
    ) -> List[List[Optional[str]]]:
        """
        Remove duplicate date rows (keeping the first occurrence).

        Duplicate dates can occur when the same date appears on two PDF pages
        or when PDFs are merged from overlapping sources.
        """
        seen: set = set()
        unique = []
        dupes = 0
        for row in rows:
            date_key = str(row[0]).strip() if row[0] else ""
            if date_key in seen:
                dupes += 1
                logger.debug(f"[{source_name}] Duplicate date dropped: {date_key}")
            else:
                seen.add(date_key)
                unique.append(row)
        if dupes:
            logger.warning(f"[{source_name}] Removed {dupes} duplicate date rows.")
        return unique

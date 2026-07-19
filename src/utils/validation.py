"""
src/utils/validation.py
───────────────────────
DataFrame validation utilities: schema checks, referential integrity,
impossible-value detection, and quality scoring.

Usage
-----
    from src.utils.validation import validate_dataframe, check_referential_integrity
    report = validate_dataframe(df, table_name="admissions")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Result objects ────────────────────────────────────────────────────────────

@dataclass
class ColumnQuality:
    """Quality statistics for a single column."""
    name: str
    dtype: str
    n_missing: int
    pct_missing: float
    n_unique: int
    n_duplicated_rows: int = 0
    has_impossible_values: bool = False
    impossible_value_count: int = 0
    notes: List[str] = field(default_factory=list)


@dataclass
class TableQuality:
    """Aggregate quality report for an entire DataFrame."""
    table_name: str
    n_rows: int
    n_cols: int
    n_duplicate_rows: int
    memory_mb: float
    columns: List[ColumnQuality] = field(default_factory=list)
    referential_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def pct_duplicate_rows(self) -> float:
        return 100.0 * self.n_duplicate_rows / max(self.n_rows, 1)

    def overall_missing_pct(self) -> float:
        total = self.n_rows * self.n_cols
        missing = sum(c.n_missing for c in self.columns)
        return 100.0 * missing / max(total, 1)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert column-level quality metrics to a DataFrame for reporting."""
        records = []
        for col in self.columns:
            records.append({
                "column": col.name,
                "dtype": col.dtype,
                "n_missing": col.n_missing,
                "pct_missing": round(col.pct_missing, 2),
                "n_unique": col.n_unique,
                "has_impossible_values": col.has_impossible_values,
                "impossible_count": col.impossible_value_count,
                "notes": "; ".join(col.notes),
            })
        return pd.DataFrame(records)

    def summary_dict(self) -> dict:
        return {
            "table": self.table_name,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "n_duplicate_rows": self.n_duplicate_rows,
            "pct_duplicate_rows": round(self.pct_duplicate_rows, 2),
            "overall_missing_pct": round(self.overall_missing_pct(), 2),
            "memory_mb": round(self.memory_mb, 2),
            "warnings": self.warnings,
            "referential_issues": self.referential_issues,
        }


# ── Core validation ────────────────────────────────────────────────────────────

def validate_dataframe(
    df: pd.DataFrame,
    table_name: str = "unknown",
    id_cols: Optional[List[str]] = None,
) -> TableQuality:
    """
    Compute comprehensive quality metrics for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    table_name : str
    id_cols : list[str], optional
        Primary key columns used to flag duplicate records.

    Returns
    -------
    TableQuality
    """
    from src.utils.io_utils import get_memory_mb

    n_rows, n_cols = df.shape
    memory_mb = get_memory_mb(df)

    # Duplicate rows
    if id_cols:
        valid_id_cols = [c for c in id_cols if c in df.columns]
        n_dup = int(df.duplicated(subset=valid_id_cols).sum()) if valid_id_cols else int(df.duplicated().sum())
    else:
        n_dup = int(df.duplicated().sum())

    tq = TableQuality(
        table_name=table_name,
        n_rows=n_rows,
        n_cols=n_cols,
        n_duplicate_rows=n_dup,
        memory_mb=memory_mb,
    )

    if n_dup > 0:
        tq.warnings.append(f"{n_dup:,} duplicate rows detected ({100*n_dup/max(n_rows,1):.1f}%)")
        log.warning("[%s] %d duplicate rows (%.1f%%)", table_name, n_dup, 100*n_dup/max(n_rows,1))

    # Per-column quality
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        pct_missing = 100.0 * n_missing / max(n_rows, 1)
        n_unique = int(series.nunique(dropna=True))

        cq = ColumnQuality(
            name=col,
            dtype=str(series.dtype),
            n_missing=n_missing,
            pct_missing=round(pct_missing, 2),
            n_unique=n_unique,
        )

        if pct_missing > 70:
            cq.notes.append(f"HIGH MISSINGNESS ({pct_missing:.1f}%)")
        if pct_missing == 100:
            tq.warnings.append(f"Column '{col}' is 100% missing")

        tq.columns.append(cq)

    log.info(
        "[%s] rows=%d, cols=%d, dup_rows=%d, mem=%.1f MB, overall_missing=%.1f%%",
        table_name, n_rows, n_cols, n_dup, memory_mb, tq.overall_missing_pct(),
    )
    return tq


# ── Referential integrity ─────────────────────────────────────────────────────

def check_referential_integrity(
    child: pd.DataFrame,
    parent: pd.DataFrame,
    child_col: str,
    parent_col: str,
    child_name: str = "child",
    parent_name: str = "parent",
) -> Tuple[int, List[str]]:
    """
    Check that every non-null value in ``child[child_col]`` exists in
    ``parent[parent_col]``.

    Parameters
    ----------
    child, parent : pd.DataFrame
    child_col : str
        Foreign key column in the child table.
    parent_col : str
        Primary key column in the parent table.
    child_name, parent_name : str
        Human-readable names for logging.

    Returns
    -------
    Tuple[int, List[str]]
        (n_orphan_rows, list_of_issue_messages)
    """
    issues: List[str] = []

    if child_col not in child.columns:
        msg = f"Column '{child_col}' not in {child_name}"
        log.warning(msg)
        return 0, [msg]

    if parent_col not in parent.columns:
        msg = f"Column '{parent_col}' not in {parent_name}"
        log.warning(msg)
        return 0, [msg]

    valid_ids = set(parent[parent_col].dropna().unique())
    child_ids = child[child_col].dropna()
    orphans = child_ids[~child_ids.isin(valid_ids)]
    n_orphan = len(orphans)

    if n_orphan > 0:
        pct = 100.0 * n_orphan / max(len(child), 1)
        msg = (
            f"Referential integrity: {n_orphan:,} rows in {child_name}.{child_col} "
            f"have no match in {parent_name}.{parent_col} ({pct:.2f}%)"
        )
        log.warning(msg)
        issues.append(msg)
    else:
        log.info("Referential integrity OK: %s.%s → %s.%s", child_name, child_col, parent_name, parent_col)

    return n_orphan, issues


# ── Impossible value detection ────────────────────────────────────────────────

# Table-level impossible value rules
# Each entry: column_name → (min_valid, max_valid, description)
_IMPOSSIBLE_VALUE_RULES: Dict[str, Dict[str, Tuple[float, float, str]]] = {
    "admissions": {
        "los_hours": (0, 365 * 24, "LoS must be 0–8760 hours"),
    },
    "patients": {
        "anchor_age": (0, 130, "Age must be 0–130"),
    },
    "icustays": {
        "los": (0, 365, "ICU LoS must be 0–365 days"),
    },
    "labevents": {
        "valuenum": (-1e6, 1e9, "Lab value must be within plausible range"),
        "ref_range_lower": (-1e6, 1e9, "Reference range plausibility"),
        "ref_range_upper": (-1e6, 1e9, "Reference range plausibility"),
    },
    "chartevents": {
        "valuenum": (-1e6, 1e9, "Chart value must be within plausible range"),
    },
    "outputevents": {
        "value": (0, 1e6, "Output volume must be non-negative"),
    },
    "inputevents": {
        "amount": (0, 1e7, "Infusion amount must be non-negative"),
        "rate": (0, 1e5, "Infusion rate must be non-negative"),
        "patientweight": (1, 500, "Patient weight must be 1–500 kg"),
    },
}

# Vital-specific range rules for chartevents
VITAL_RANGES: Dict[str, Tuple[float, float]] = {
    "heart_rate":    (0, 300),
    "sbp":           (0, 300),
    "dbp":           (0, 250),
    "resp_rate":     (0, 100),
    "temperature_f": (80, 120),
    "temperature_c": (25, 45),
    "spo2":          (50, 100),
}


def detect_impossible_values(
    df: pd.DataFrame,
    table_name: str,
) -> Dict[str, int]:
    """
    Check numeric columns for clinically impossible values.

    Parameters
    ----------
    df : pd.DataFrame
    table_name : str
        Must match a key in ``_IMPOSSIBLE_VALUE_RULES``.

    Returns
    -------
    Dict[str, int]
        Mapping of column_name → count_of_impossible_values.
    """
    rules = _IMPOSSIBLE_VALUE_RULES.get(table_name, {})
    results: Dict[str, int] = {}

    for col, (lo, hi, desc) in rules.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        mask = series.notna() & ((series < lo) | (series > hi))
        count = int(mask.sum())
        if count > 0:
            log.warning(
                "[%s] Column '%s': %d impossible values (rule: %s, range=[%g, %g])",
                table_name, col, count, desc, lo, hi,
            )
        results[col] = count

    return results


def check_timestamp_consistency(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    table_name: str = "unknown",
) -> int:
    """
    Flag rows where ``end_col`` < ``start_col`` (negative duration).

    Returns
    -------
    int
        Number of rows with negative duration.
    """
    if start_col not in df.columns or end_col not in df.columns:
        return 0

    start = pd.to_datetime(df[start_col], errors="coerce")
    end = pd.to_datetime(df[end_col], errors="coerce")
    n_neg = int((end < start).sum())

    if n_neg > 0:
        log.warning(
            "[%s] %d rows with %s < %s (negative duration)",
            table_name, n_neg, end_col, start_col,
        )
    return n_neg

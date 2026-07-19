"""
src/utils/schema.py
───────────────────
Automatic schema inference for MIMIC-IV tables: dtypes, date columns,
categorical columns, and numerical columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger(__name__)

DATE_KEYWORDS = ("time", "date", "dod", "intime", "outtime")
ID_KEYWORDS = ("_id", "id", "seq_num")


@dataclass
class TableSchema:
    """Inferred schema for a single table."""

    table_name: str
    n_rows: int
    n_cols: int
    columns: List[str] = field(default_factory=list)
    dtypes: Dict[str, str] = field(default_factory=dict)
    date_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    id_columns: List[str] = field(default_factory=list)
    text_columns: List[str] = field(default_factory=list)
    memory_mb: float = 0.0

    def to_dict(self) -> dict:
        return {
            "table": self.table_name,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "date_columns": self.date_columns,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "id_columns": self.id_columns,
            "text_columns": self.text_columns,
            "dtypes": self.dtypes,
            "memory_mb": round(self.memory_mb, 2),
        }


def _looks_like_date(series: pd.Series, sample_size: int = 500) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype != object:
        return False
    sample = series.dropna().astype(str).head(sample_size)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.8


def infer_table_schema(
    df: pd.DataFrame,
    table_name: str,
    config_date_cols: Optional[List[str]] = None,
    config_cat_cols: Optional[List[str]] = None,
    config_id_cols: Optional[List[str]] = None,
    text_col: Optional[str] = None,
) -> TableSchema:
    """
    Infer column roles for a loaded DataFrame.

    Config hints take precedence over heuristics.
    """
    from src.utils.io_utils import get_memory_mb

    config_date_cols = config_date_cols or []
    config_cat_cols = config_cat_cols or []
    config_id_cols = config_id_cols or []

    date_cols: Set[str] = set(config_date_cols)
    cat_cols: Set[str] = set(config_cat_cols)
    id_cols: Set[str] = set(config_id_cols)
    num_cols: Set[str] = set()
    text_cols: Set[str] = set()

    if text_col and text_col in df.columns:
        text_cols.add(text_col)

    for col in df.columns:
        lower = col.lower()
        series = df[col]

        if col in config_id_cols or any(k in lower for k in ID_KEYWORDS):
            id_cols.add(col)

        if col in config_date_cols or any(k in lower for k in DATE_KEYWORDS):
            if _looks_like_date(series) or col in config_date_cols:
                date_cols.add(col)

        if pd.api.types.is_numeric_dtype(series):
            num_cols.add(col)
        elif pd.api.types.is_datetime64_any_dtype(series):
            date_cols.add(col)
        elif col in config_cat_cols:
            cat_cols.add(col)
        elif series.dtype == object or pd.api.types.is_categorical_dtype(series):
            n_unique = series.nunique(dropna=True)
            n_rows = max(len(series), 1)
            if col in text_cols or (n_unique > 50 and series.astype(str).str.len().mean() > 20):
                text_cols.add(col)
            elif n_unique <= min(100, int(0.05 * n_rows) + 1):
                cat_cols.add(col)
            else:
                coerced = pd.to_numeric(series, errors="coerce")
                if coerced.notna().mean() > 0.9:
                    num_cols.add(col)
                else:
                    cat_cols.add(col)

    assigned = date_cols | cat_cols | num_cols | text_cols | id_cols
    for col in df.columns:
        if col not in assigned and pd.api.types.is_numeric_dtype(df[col]):
            num_cols.add(col)

    schema = TableSchema(
        table_name=table_name,
        n_rows=len(df),
        n_cols=df.shape[1],
        columns=list(df.columns),
        dtypes={c: str(df[c].dtype) for c in df.columns},
        date_columns=sorted(date_cols),
        categorical_columns=sorted(cat_cols),
        numerical_columns=sorted(num_cols),
        id_columns=sorted(id_cols),
        text_columns=sorted(text_cols),
        memory_mb=get_memory_mb(df),
    )
    log.info(
        "Schema [%s]: dates=%d, cat=%d, num=%d, ids=%d, text=%d",
        table_name,
        len(schema.date_columns),
        len(schema.categorical_columns),
        len(schema.numerical_columns),
        len(schema.id_columns),
        len(schema.text_columns),
    )
    return schema


def schemas_to_dataframe(schemas: Dict[str, TableSchema]) -> pd.DataFrame:
    """Flatten multiple table schemas into a summary DataFrame."""
    rows = []
    for name, schema in schemas.items():
        row = schema.to_dict()
        row["table"] = name
        for key in ("date_columns", "categorical_columns", "numerical_columns", "id_columns", "text_columns"):
            if key in row and isinstance(row[key], list):
                row[key] = ", ".join(row[key])
        if "dtypes" in row and isinstance(row["dtypes"], dict):
            row["dtypes"] = str(row["dtypes"])
        rows.append(row)
    return pd.DataFrame(rows)

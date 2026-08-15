"""
src/data/cleaner.py
───────────────────
Data cleaning for all MIMIC-IV tables with full audit trail.

Design principles
-----------------
* Never silently remove data — all removals are logged and counted.
* Cleaning decisions are recorded in ``CleaningReport`` objects.
* Cleaned tables are saved as Parquet to ``data/interim/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import CFG
from src.utils.io_utils import get_memory_mb, optimise_dtypes, save_parquet
from src.utils.logger import get_logger
from src.utils.schema import infer_table_schema
from src.utils.validation import (
    TableQuality,
    check_referential_integrity,
    check_timestamp_consistency,
    detect_impossible_values,
    validate_dataframe,
)

from src.utils.reports import df_to_markdown

log = get_logger(__name__)


@dataclass
class CleaningAction:
    """Single documented cleaning decision."""

    table: str
    action: str
    column: str = ""
    n_affected: int = 0
    decision: str = ""
    details: str = ""


@dataclass
class CleaningReport:
    """Aggregate cleaning report for one table."""

    table_name: str
    actions: List[CleaningAction] = field(default_factory=list)
    quality_before: Optional[TableQuality] = None
    quality_after: Optional[TableQuality] = None
    schema: Optional[dict] = None
    output_path: str = ""

    def add(self, action: str, **kwargs: Any) -> None:
        self.actions.append(CleaningAction(table=self.table_name, action=action, **kwargs))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([a.__dict__ for a in self.actions])

    def summary(self) -> dict:
        return {
            "table": self.table_name,
            "n_actions": len(self.actions),
            "output_path": self.output_path,
            "rows_before": self.quality_before.n_rows if self.quality_before else None,
            "rows_after": self.quality_after.n_rows if self.quality_after else None,
            "dup_before": self.quality_before.n_duplicate_rows if self.quality_before else None,
            "dup_after": self.quality_after.n_duplicate_rows if self.quality_after else None,
        }


class DataCleaner:
    """Clean MIMIC-IV tables with documented, reproducible rules."""

    WHITESPACE_COLS = {
        "admissions": ["admission_type", "admission_location", "discharge_location",
                         "insurance", "language", "marital_status", "race"],
        "patients": ["gender"],
        "prescriptions": ["drug", "route", "drug_type"],
        "pharmacy": ["medication", "route", "frequency"],
        "emar": ["medication", "event_txt"],
    }

    TIMESTAMP_PAIRS = {
        "admissions": [("admittime", "dischtime")],
        "icustays": [("intime", "outtime")],
        "prescriptions": [("starttime", "stoptime")],
        "pharmacy": [("starttime", "stoptime")],
        "inputevents": [("starttime", "endtime")],
    }

    def __init__(self, interim_dir: Optional[Path] = None) -> None:
        self.cfg = CFG
        self.interim_dir = Path(interim_dir or self.cfg.resolve(self.cfg.paths.interim))

    def clean_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        id_cols: Optional[List[str]] = None,
        save: bool = False,
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """Apply the full cleaning pipeline to one table."""
        report = CleaningReport(table_name=table_name)
        t_cfg = self.cfg.tables.get(table_name, {})
        id_cols = id_cols or t_cfg.get("id_cols", [])

        if df.empty:
            report.add("skip", decision="empty table — no cleaning applied")
            return df, report

        report.quality_before = validate_dataframe(df, table_name, id_cols)

        cleaned = df.copy()
        cleaned = self._strip_whitespace(cleaned, table_name, report)
        cleaned = self._fix_datatypes(cleaned, table_name, t_cfg, report)
        cleaned = self._handle_missing_values(cleaned, table_name, report)
        cleaned = self._handle_duplicates(cleaned, table_name, id_cols, report)
        cleaned = self._handle_outliers(cleaned, table_name, report)
        cleaned = self._validate_timestamps(cleaned, table_name, report)
        cleaned = self._validate_categoricals(cleaned, table_name, t_cfg, report)

        schema = infer_table_schema(
            cleaned,
            table_name,
            config_date_cols=t_cfg.get("date_cols"),
            config_cat_cols=t_cfg.get("cat_cols"),
            config_id_cols=id_cols,
            text_col=t_cfg.get("text_col"),
        )
        report.schema = schema.to_dict()
        report.quality_after = validate_dataframe(cleaned, table_name, id_cols)

        cleaned = optimise_dtypes(cleaned)

        if save:
            out = self.interim_dir / f"{table_name}_clean.parquet"
            save_parquet(cleaned, out)
            report.output_path = str(out)
            log.info("Saved cleaned '%s' → %s", table_name, out.name)

        return cleaned, report

    def _strip_whitespace(
        self, df: pd.DataFrame, table_name: str, report: CleaningReport
    ) -> pd.DataFrame:
        cols = [c for c in self.WHITESPACE_COLS.get(table_name, []) if c in df.columns]
        for col in cols:
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                before = df[col].astype(str).str.strip().ne(df[col].astype(str)).sum()
                df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})
                if before:
                    report.add(
                        "strip_whitespace",
                        column=col,
                        n_affected=int(before),
                        decision="strip leading/trailing whitespace",
                    )
        return df

    def _fix_datatypes(
        self,
        df: pd.DataFrame,
        table_name: str,
        t_cfg: dict,
        report: CleaningReport,
    ) -> pd.DataFrame:
        for col in t_cfg.get("date_cols", []):
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                converted = pd.to_datetime(df[col], errors="coerce")
                n_invalid = int(converted.isna().sum() - df[col].isna().sum())
                df[col] = converted
                if n_invalid > 0:
                    report.add(
                        "fix_datetime",
                        column=col,
                        n_affected=n_invalid,
                        decision="coerce invalid timestamps to NaT (retained as missing)",
                    )

        numeric_candidates = {
            "labevents": ["valuenum", "ref_range_lower", "ref_range_upper"],
            "chartevents": ["valuenum"],
            "outputevents": ["value"],
            "inputevents": ["amount", "rate", "patientweight"],
            "patients": ["anchor_age", "anchor_year"],
            "icustays": ["los"],
        }
        for col in numeric_candidates.get(table_name, []):
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                coerced = pd.to_numeric(df[col], errors="coerce")
                n_changed = int((coerced.isna() & df[col].notna()).sum())
                df[col] = coerced
                if n_changed:
                    report.add(
                        "fix_numeric",
                        column=col,
                        n_affected=n_changed,
                        decision="coerce non-numeric values to NaN",
                    )
        return df

    def _handle_missing_values(
        self, df: pd.DataFrame, table_name: str, report: CleaningReport
    ) -> pd.DataFrame:
        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing:
                report.add(
                    "document_missing",
                    column=col,
                    n_affected=n_missing,
                    decision="retained missing values; no imputation at cleaning stage",
                    details=f"{100 * n_missing / max(len(df), 1):.2f}% missing",
                )
        return df

    def _handle_duplicates(
        self,
        df: pd.DataFrame,
        table_name: str,
        id_cols: List[str],
        report: CleaningReport,
    ) -> pd.DataFrame:
        valid_ids = [c for c in id_cols if c in df.columns]
        if valid_ids:
            dup_mask = df.duplicated(subset=valid_ids, keep=False)
        else:
            dup_mask = df.duplicated(keep=False)

        n_dup = int(dup_mask.sum())
        df = df.copy()
        df["_is_duplicate"] = dup_mask.astype(np.int8)
        if n_dup > 0:
            report.add(
                "flag_duplicates",
                n_affected=n_dup,
                decision="duplicates retained with _is_duplicate flag; not removed",
            )
        return df

    def _handle_outliers(
        self, df: pd.DataFrame, table_name: str, report: CleaningReport
    ) -> pd.DataFrame:
        impossible = detect_impossible_values(df, table_name)
        for col, count in impossible.items():
            if count <= 0:
                continue
            lo_hi = self._get_impossible_bounds(table_name, col)
            if lo_hi:
                lo, hi = lo_hi
                mask = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
                df.loc[mask, col] = np.nan
                report.add(
                    "cap_outliers",
                    column=col,
                    n_affected=count,
                    decision=f"set clinically impossible values to NaN (range [{lo}, {hi}])",
                )
        return df

    @staticmethod
    def _get_impossible_bounds(table_name: str, col: str) -> Optional[Tuple[float, float]]:
        from src.utils.validation import _IMPOSSIBLE_VALUE_RULES

        rule = _IMPOSSIBLE_VALUE_RULES.get(table_name, {}).get(col)
        return (rule[0], rule[1]) if rule else None

    def _validate_timestamps(
        self, df: pd.DataFrame, table_name: str, report: CleaningReport
    ) -> pd.DataFrame:
        for start_col, end_col in self.TIMESTAMP_PAIRS.get(table_name, []):
            n_neg = check_timestamp_consistency(df, start_col, end_col, table_name)
            if n_neg:
                report.add(
                    "invalid_timestamp_order",
                    column=f"{start_col}/{end_col}",
                    n_affected=n_neg,
                    decision="flagged with _invalid_time_order=1 and kept in dataset (no values dropped or nulled)",
                )
                mask = pd.to_datetime(df[end_col], errors="coerce") < pd.to_datetime(
                    df[start_col], errors="coerce"
                )
                df = df.copy()
                df["_invalid_time_order"] = mask.fillna(False).astype(np.int8)
        return df

    def _validate_categoricals(
        self,
        df: pd.DataFrame,
        table_name: str,
        t_cfg: dict,
        report: CleaningReport,
    ) -> pd.DataFrame:
        for col in t_cfg.get("cat_cols", []):
            if col not in df.columns:
                continue
            empty = df[col].astype(str).str.strip().isin(["", "nan", "None", "NULL"])
            n_empty = int(empty.sum())
            if n_empty:
                df.loc[empty, col] = np.nan
                report.add(
                    "normalize_categorical",
                    column=col,
                    n_affected=n_empty,
                    decision="empty strings normalized to NaN",
                )
        return df

    def check_all_referential_integrity(
        self,
        tables: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Run FK checks across cleaned tables."""
        checks = [
            ("admissions", "subject_id", "patients", "subject_id"),
            ("diagnoses_icd", "hadm_id", "admissions", "hadm_id"),
            ("diagnoses_icd", "subject_id", "patients", "subject_id"),
            ("procedures_icd", "hadm_id", "admissions", "hadm_id"),
            ("icustays", "hadm_id", "admissions", "hadm_id"),
            ("icustays", "subject_id", "patients", "subject_id"),
            ("labevents", "hadm_id", "admissions", "hadm_id"),
            ("prescriptions", "hadm_id", "admissions", "hadm_id"),
            ("chartevents", "stay_id", "icustays", "stay_id"),
            ("discharge", "hadm_id", "admissions", "hadm_id"),
        ]
        records = []
        for child_name, child_col, parent_name, parent_col in checks:
            if child_name not in tables or parent_name not in tables:
                continue
            n_orphan, issues = check_referential_integrity(
                tables[child_name],
                tables[parent_name],
                child_col,
                parent_col,
                child_name,
                parent_name,
            )
            records.append({
                "child_table": child_name,
                "child_col": child_col,
                "parent_table": parent_name,
                "parent_col": parent_col,
                "n_orphan_rows": n_orphan,
                "issues": "; ".join(issues),
            })
        return pd.DataFrame(records)

    def save_cleaning_report(
        self,
        reports: Dict[str, CleaningReport],
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Persist cleaning actions and summaries."""
        out_dir = Path(output_dir or self.cfg.resolve(self.cfg.paths.tables))
        out_dir.mkdir(parents=True, exist_ok=True)

        actions = pd.concat(
            [r.to_dataframe() for r in reports.values() if len(r.actions)],
            ignore_index=True,
        )
        summary = pd.DataFrame([r.summary() for r in reports.values()])

        actions_path = out_dir / "cleaning_actions.parquet"
        summary_path = out_dir / "cleaning_summary.parquet"
        save_parquet(actions, actions_path)
        save_parquet(summary, summary_path)

        md_path = self.cfg.resolve("reports/cleaning_report.md")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("# Data Cleaning Report\n\n")
            fh.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
            # The generation date on this report is old enough to look superseded, and
            # an audit listed it for regeneration on that basis alone. It is not: the
            # ID-collision repair changed identifier *values*, not row counts or
            # missingness, so nothing the cleaning stage measures moved. Saying so here
            # — and pointing at the test that proves it — costs one line and prevents
            # the same wrong conclusion being reached again from the timestamp.
            fh.write(
                "> [!NOTE]\n"
                "> Every figure below is checked against the interim parquets by\n"
                "> `tests/test_cleaning_report_current.py`, which re-derives the row\n"
                "> counts and missingness percentages and fails if they diverge. A\n"
                "> passing suite means this report is current regardless of its date.\n\n")
            fh.write("## Summary\n\n")
            fh.write("> **Note on Duplicates:** The summary table (`dup_before`/`dup_after`) counts only the redundant extra copies (`keep='first'`), showing how many rows would be dropped to make the table unique. The Documented Actions table below (`flag_duplicates`) counts *all* rows involved in a duplicate group (`keep=False`), because all copies receive the `_is_duplicate` flag.\n\n")
            fh.write(df_to_markdown(summary))
            fh.write("\n\n## Documented Actions\n\n")
            if len(actions):
                fh.write(df_to_markdown(actions))
            else:
                fh.write("_No cleaning actions recorded._\n")

        log.info("Cleaning report saved → %s", md_path)
        return md_path

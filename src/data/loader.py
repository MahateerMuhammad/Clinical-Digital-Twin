"""
src/data/loader.py
──────────────────
DataLoader class — the single entry point for reading all MIMIC-IV tables.

Design
------
* Every table has a dedicated ``load_<table>()`` method that applies the
  correct date columns, dtype hints, and size-aware reading strategy (full
  read vs. chunked).
* After loading, a ``TableSummary`` is printed to the logger and returned to
  the caller alongside the DataFrame.
* Large tables (marked ``large: true`` in config.yaml) are read in chunks.

Usage
-----
    from src.data.loader import DataLoader
    loader = DataLoader()
    patients, summary = loader.load_patients()
    admissions, summary = loader.load_admissions()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.config import CFG
from src.utils.io_utils import (
    get_memory_mb,
    optimise_dtypes,
    read_csv_chunked,
    read_csv_full,
)
from src.utils.logger import get_logger
from src.utils.validation import validate_dataframe, TableQuality

log = get_logger(__name__)


# ── TableSummary ──────────────────────────────────────────────────────────────

@dataclass
class TableSummary:
    """Lightweight metadata returned by every load_* call."""
    table_name: str
    n_rows: int
    n_cols: int
    columns: List[str] = field(default_factory=list)
    dtypes: Dict[str, str] = field(default_factory=dict)
    memory_mb: float = 0.0
    load_time_sec: float = 0.0
    quality: Optional[TableQuality] = None
    is_partial: bool = False
    partial_note: str = ""

    def __str__(self) -> str:
        lines = [
            f"── {self.table_name} ──────────────────────────────",
            f"  Rows      : {self.n_rows:,}",
            f"  Columns   : {self.n_cols}",
            f"  Memory    : {self.memory_mb:.1f} MB",
            f"  Load time : {self.load_time_sec:.1f} s",
        ]
        if self.is_partial:
            lines.append(f"  ⚠  PARTIAL : {self.partial_note}")
        return "\n".join(lines)


# ── DataLoader class ──────────────────────────────────────────────────────────

class DataLoader:
    """
    Loads all MIMIC-IV tables from the raw data directories.

    Attributes
    ----------
    cfg : Config
        Singleton config object.
    validate : bool
        Whether to run ``validate_dataframe`` after each load.
    optimise : bool
        Whether to run dtype optimisation after each load.
    """

    def __init__(
        self,
        validate: bool = True,
        optimise: bool = True,
    ) -> None:
        self.cfg = CFG
        self.validate = validate
        self.optimise = optimise

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load(
        self,
        table_name: str,
        usecols: Optional[List[str]] = None,
        max_chunks: Optional[int] = None,
        row_processor: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ) -> Tuple[pd.DataFrame, TableSummary]:
        """
        Generic loader that delegates to chunked or full read based on the
        ``large`` flag in config.
        """
        t_cfg = self.cfg.tables[table_name]
        filepath = self.cfg.table_file(table_name)
        date_cols: List[str] = t_cfg.get("date_cols", [])
        id_cols: List[str] = t_cfg.get("id_cols", [])
        is_large: bool = t_cfg.get("large", False)
        is_partial: bool = "PARTIAL" in t_cfg.get("note", "")
        partial_note: str = t_cfg.get("note", "")

        if not filepath.exists():
            log.error("File NOT FOUND: %s — skipping.", filepath)
            empty = pd.DataFrame()
            return empty, TableSummary(
                table_name=table_name,
                n_rows=0, n_cols=0,
                is_partial=True,
                partial_note=f"File not found: {filepath}",
            )

        start = time.time()

        if is_large:
            df = read_csv_chunked(
                filepath=filepath,
                date_cols=date_cols,
                usecols=usecols,
                max_chunks=max_chunks,
                row_processor=row_processor,
            )
        else:
            df = read_csv_full(
                filepath=filepath,
                date_cols=date_cols,
                usecols=usecols,
            )

        if self.optimise and len(df) > 0:
            df = optimise_dtypes(df)

        elapsed = time.time() - start

        quality: Optional[TableQuality] = None
        if self.validate and len(df) > 0:
            quality = validate_dataframe(df, table_name=table_name, id_cols=id_cols)

        summary = TableSummary(
            table_name=table_name,
            n_rows=len(df),
            n_cols=df.shape[1],
            columns=list(df.columns),
            dtypes={c: str(df[c].dtype) for c in df.columns},
            memory_mb=get_memory_mb(df),
            load_time_sec=elapsed,
            quality=quality,
            is_partial=is_partial,
            partial_note=partial_note,
        )

        log.info("Loaded table '%s': %s", table_name, summary)
        return df, summary

    # ── hosp ─────────────────────────────────────────────────────────────────

    def load_patients(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load patients.csv from hosp."""
        return self._load("patients", **kwargs)

    def load_admissions(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load admissions.csv from hosp."""
        return self._load("admissions", **kwargs)

    def load_diagnoses_icd(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load diagnoses_icd.csv from hosp."""
        return self._load("diagnoses_icd", **kwargs)

    def load_d_icd_diagnoses(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load d_icd_diagnoses.csv from hosp (diagnosis code dictionary)."""
        return self._load("d_icd_diagnoses", **kwargs)

    def load_procedures_icd(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load procedures_icd.csv from hosp."""
        return self._load("procedures_icd", **kwargs)

    def load_d_icd_procedures(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load d_icd_procedures.csv from hosp (procedure code dictionary)."""
        return self._load("d_icd_procedures", **kwargs)

    def load_labevents(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load labevents.csv from hosp."""
        # Filter to only relevant lab itemids to save memory
        key_itemids = set()
        for v in CFG.key_labs.values():
            if isinstance(v, list):
                key_itemids.update(v)
            else:
                key_itemids.add(v)
        
        def filter_labs(chunk: pd.DataFrame) -> pd.DataFrame:
            if "itemid" in chunk.columns:
                return chunk[chunk["itemid"].isin(key_itemids)]
            return chunk

        kwargs["row_processor"] = filter_labs
        return self._load("labevents", **kwargs)

    def load_d_labitems(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load d_labitems.csv from hosp (lab item dictionary)."""
        return self._load("d_labitems", **kwargs)

    def load_prescriptions(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load prescriptions.csv from hosp."""
        return self._load("prescriptions", **kwargs)

    def load_pharmacy(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load pharmacy.csv from hosp."""
        return self._load("pharmacy", **kwargs)

    def load_emar(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load emar.csv from hosp (electronic medication administration record)."""
        return self._load("emar", **kwargs)

    def load_emar_detail(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load emar_detail.csv from hosp."""
        return self._load("emar_detail", **kwargs)

    # ── icu ──────────────────────────────────────────────────────────────────

    def load_icustays(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load icustays.csv from icu."""
        return self._load("icustays", **kwargs)

    def load_chartevents(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load chartevents.csv from icu."""
        # Filter to only relevant vital itemids to save memory
        vital_itemids = set()
        for v in CFG.vitals.values():
            if isinstance(v, list):
                vital_itemids.update(v)
            else:
                vital_itemids.add(v)
                
        def filter_vitals(chunk: pd.DataFrame) -> pd.DataFrame:
            if "itemid" in chunk.columns:
                return chunk[chunk["itemid"].isin(vital_itemids)]
            return chunk

        kwargs["row_processor"] = filter_vitals
        return self._load("chartevents", **kwargs)

    def load_d_items(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load d_items.csv from icu (item dictionary)."""
        return self._load("d_items", **kwargs)

    def load_inputevents(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load inputevents.csv from icu."""
        return self._load("inputevents", **kwargs)

    def load_outputevents(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load outputevents.csv from icu."""
        return self._load("outputevents", **kwargs)

    # ── ed ───────────────────────────────────────────────────────────────────

    def load_edstays(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load edstays.csv from the ED module (ED stay → hadm_id link table)."""
        return self._load("edstays", **kwargs)

    def load_triage(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load triage.csv from the ED module (first-contact vitals)."""
        return self._load("triage", **kwargs)

    def load_ed_vitalsign(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """
        Load vitalsign.csv from the ED module (serial vitals).

        Named `ed_vitalsign` rather than `vitalsign` because `vitals` already
        denotes the ICU chartevents-derived feature set; two things called vitals
        in one namespace is how the wrong one gets wired up.
        """
        return self._load("ed_vitalsign", **kwargs)

    def load_medrecon(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load medrecon.csv from the ED module (home-medication reconciliation)."""
        return self._load("medrecon", **kwargs)

    # ── notes ────────────────────────────────────────────────────────────────

    def load_discharge(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load discharge.csv from notes (discharge summaries)."""
        return self._load("discharge", **kwargs)

    def load_radiology(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load radiology.csv from notes (if available)."""
        if not self.cfg.table_file("radiology").exists():
            log.warning("radiology.csv not found — skipping.")
            return pd.DataFrame(), TableSummary(
                table_name="radiology", n_rows=0, n_cols=0,
                is_partial=True, partial_note="File not found",
            )
        return self._load("radiology", **kwargs)

    def load_radiology_detail(self, **kwargs) -> Tuple[pd.DataFrame, TableSummary]:
        """Load radiology_detail.csv from notes."""
        return self._load("radiology_detail", **kwargs)

    # ── Bulk helpers ─────────────────────────────────────────────────────────

    def load_all_small_tables(self) -> Dict[str, Tuple[pd.DataFrame, TableSummary]]:
        """
        Load all non-large tables in one call.

        Returns
        -------
        Dict[str, Tuple[pd.DataFrame, TableSummary]]
        """
        loaders = {
            "patients": self.load_patients,
            "admissions": self.load_admissions,
            "diagnoses_icd": self.load_diagnoses_icd,
            "d_icd_diagnoses": self.load_d_icd_diagnoses,
            "procedures_icd": self.load_procedures_icd,
            "d_icd_procedures": self.load_d_icd_procedures,
            "d_labitems": self.load_d_labitems,
            "icustays": self.load_icustays,
            "d_items": self.load_d_items,
        }

        results: Dict[str, Tuple[pd.DataFrame, TableSummary]] = {}
        for name, fn in loaders.items():
            try:
                results[name] = fn()
            except Exception as exc:   # noqa: BLE001
                log.error("Failed to load '%s': %s", name, exc)

        return results

    def generate_dataset_summary(
        self,
        summaries: Dict[str, TableSummary],
    ) -> pd.DataFrame:
        """
        Create a summary DataFrame of all loaded tables.

        Parameters
        ----------
        summaries : dict
            Mapping of table_name → TableSummary.

        Returns
        -------
        pd.DataFrame
        """
        records = []
        for name, s in summaries.items():
            records.append({
                "table": name,
                "n_rows": s.n_rows,
                "n_cols": s.n_cols,
                "memory_mb": round(s.memory_mb, 1),
                "load_time_sec": round(s.load_time_sec, 1),
                "is_partial": s.is_partial,
                "partial_note": s.partial_note,
                "overall_missing_pct": (
                    round(s.quality.overall_missing_pct(), 1)
                    if s.quality else None
                ),
                "duplicate_rows": (
                    s.quality.n_duplicate_rows if s.quality else None
                ),
            })
        return pd.DataFrame(records).sort_values("memory_mb", ascending=False)

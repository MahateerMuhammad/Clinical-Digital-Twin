"""
src/features/procedure.py
─────────────────────────
Procedure feature engineering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

MAJOR_PROCEDURE_PREFIXES = ("0", "1", "2", "3", "4", "5")


def build_procedure_features(procedures: pd.DataFrame) -> pd.DataFrame:
    """Admission-level procedure features."""
    if procedures.empty:
        return pd.DataFrame(columns=["hadm_id"])

    proc = procedures.copy()
    proc["icd_code"] = proc["icd_code"].astype(str)

    agg = proc.groupby("hadm_id", observed=True).agg(
        procedure_count=("icd_code", "count"),
        unique_procedure_count=("icd_code", "nunique"),
    ).reset_index()

    def major_count(group: pd.DataFrame) -> int:
        codes = group["icd_code"].astype(str)
        return int(codes.str.startswith(MAJOR_PROCEDURE_PREFIXES).sum())

    major = proc.groupby("hadm_id", observed=True).apply(major_count, include_groups=False)
    agg["major_procedure_count"] = agg["hadm_id"].map(major).fillna(0).astype(np.int16)
    agg["has_major_procedure"] = (agg["major_procedure_count"] > 0).astype(np.int8)

    log.info("Procedure features: %d admissions", len(agg))
    return agg

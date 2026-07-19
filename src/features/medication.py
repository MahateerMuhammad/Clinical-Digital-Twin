"""
src/features/medication.py
────────────────────────────
Medication and pharmacy feature engineering.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils.io_utils import aggregate_chunked
from src.utils.logger import get_logger

log = get_logger(__name__)

DRUG_CLASS_KEYWORDS = {
    "med_class_antibiotic": ["vanc", "cef", "azithro", "cipro", "levo", "piperacillin", "meropenem"],
    "med_class_anticoagulant": ["heparin", "warfarin", "enoxaparin", "apixaban", "rivaroxaban"],
    "med_class_insulin": ["insulin"],
    "med_class_opioid": ["morphine", "fentanyl", "hydromorphone", "oxycodone"],
    "med_class_statin": ["statin", "atorva", "simva", "rosuva"],
    "med_class_beta_blocker": ["metoprolol", "carvedilol", "labetalol", "atenolol"],
    "med_class_ace_inhibitor": ["lisinopril", "enalapril", "captopril", "ramipril"],
}


def _drug_class_flags(drugs: pd.Series) -> Dict[str, int]:
    text = " ".join(drugs.dropna().astype(str).str.lower().tolist())
    return {col: int(any(kw in text for kw in keywords)) for col, keywords in DRUG_CLASS_KEYWORDS.items()}


def build_medication_features(prescriptions: pd.DataFrame) -> pd.DataFrame:
    """Admission-level medication features from prescriptions table."""
    if prescriptions.empty:
        return pd.DataFrame(columns=["hadm_id"])

    rx = prescriptions.copy()
    if "starttime" in rx.columns:
        rx["starttime"] = pd.to_datetime(rx["starttime"], errors="coerce")
    if "stoptime" in rx.columns:
        rx["stoptime"] = pd.to_datetime(rx["stoptime"], errors="coerce")

    durations = (rx["stoptime"] - rx["starttime"]).dt.total_seconds() / 3600.0 if "starttime" in rx.columns and "stoptime" in rx.columns else None
    if durations is not None:
        rx["_duration_hours"] = durations

    agg_dict = {
        "pharmacy_id": ["count"],
        "drug": ["nunique"]
    }
    if "_duration_hours" in rx.columns:
        agg_dict["_duration_hours"] = ["mean", "max"]

    if "drug" in rx.columns:
        drugs_lower = rx["drug"].dropna().astype(str).str.lower()
        for col, keywords in DRUG_CLASS_KEYWORDS.items():
            pattern = "|".join(keywords)
            rx[col] = drugs_lower.str.contains(pattern, regex=True).astype(np.int8)
            agg_dict[col] = ["max"]

    grouped = rx.groupby("hadm_id", observed=True).agg(agg_dict)
    grouped.columns = ["_".join(c).strip("_") for c in grouped.columns]
    
    rename_cols = {
        "pharmacy_id_count": "medication_count",
        "drug_nunique": "unique_medications",
        "_duration_hours_mean": "med_duration_hours_mean",
        "_duration_hours_max": "med_duration_hours_max",
    }
    for col in DRUG_CLASS_KEYWORDS.keys():
        rename_cols[f"{col}_max"] = col

    result = grouped.reset_index().rename(columns=rename_cols)

    log.info("Medication features: %d admissions × %d cols", len(result), result.shape[1])
    return result


def build_medication_features_chunked(filepath: str) -> pd.DataFrame:
    """Chunked medication counts per hadm_id."""
    partial = aggregate_chunked(
        filepath=filepath,
        group_col="hadm_id",
        agg_funcs={"drug": "nunique", "pharmacy_id": "count"},
        date_cols=["starttime", "stoptime"],
        usecols=["hadm_id", "drug", "pharmacy_id", "starttime", "stoptime"],
    )
    if partial.empty:
        return pd.DataFrame(columns=["hadm_id"])

    partial = partial.rename(columns={
        "drug_nunique": "unique_medications",
        "pharmacy_id_count": "medication_count",
    })
    log.info("Chunked medication features: %d admissions", len(partial))
    return partial

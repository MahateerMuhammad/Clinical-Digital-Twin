"""
src/features/diagnosis.py
───────────────────────────
Diagnosis-based feature engineering: counts, Charlson CCI, chronic flags.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger(__name__)

CHARLSON_WEIGHTS = {
    "myocardial_infarction": 1,
    "congestive_heart_failure": 1,
    "peripheral_vascular_disease": 1,
    "cerebrovascular_disease": 1,
    "dementia": 1,
    "copd": 1,
    "connective_tissue_disease": 1,
    "peptic_ulcer": 1,
    "mild_liver_disease": 1,
    "diabetes_uncomplicated": 1,
    "diabetes_complicated": 2,
    "hemiplegia_paraplegia": 2,
    "renal_disease": 2,
    "malignancy": 2,
    "severe_liver_disease": 3,
    "metastatic_tumor": 6,
    "aids": 6,
}


def _normalize_icd(code: pd.Series) -> pd.Series:
    return code.astype(str).str.upper().str.replace(".", "", regex=False)


def _match_mask(
    dx: pd.DataFrame,
    icd9: List[str],
    icd10: List[str],
) -> pd.Series:
    """Vectorized ICD prefix matching."""
    codes = _normalize_icd(dx["icd_code"])
    v9 = dx["icd_version"] == 9
    v10 = dx["icd_version"] == 10
    mask = pd.Series(False, index=dx.index)
    for pat in icd9:
        mask |= v9 & codes.str.startswith(pat.upper().replace(".", ""))
    for pat in icd10:
        mask |= v10 & codes.str.startswith(pat.upper().replace(".", ""))
    return mask


def build_charlson_flags(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Return hadm_id-level Charlson comorbidity flags and score (vectorized)."""
    hadm_ids = diagnoses[["hadm_id"]].drop_duplicates().sort_values("hadm_id")
    result = hadm_ids.copy()
    dx = diagnoses.copy()
    dx["icd_version"] = pd.to_numeric(dx["icd_version"], errors="coerce").fillna(9).astype(int)

    for condition, mappings in CFG.charlson.items():
        col = f"cci_{condition}"
        mask = _match_mask(dx, mappings.get("icd9", []), mappings.get("icd10", []))
        matched_hadm = dx.loc[mask, "hadm_id"].drop_duplicates()
        result[col] = result["hadm_id"].isin(matched_hadm).astype(np.int8)

    cci_cols = [c for c in result.columns if c.startswith("cci_")]
    result["charlson_comorbidity_index"] = sum(
        result[c] * CHARLSON_WEIGHTS.get(c.replace("cci_", ""), 1) for c in cci_cols
    ).astype(np.int16)

    log.info(
        "Charlson features: %d admissions, score range [%d, %d]",
        len(result),
        result["charlson_comorbidity_index"].min(),
        result["charlson_comorbidity_index"].max(),
    )
    return result


def build_diagnosis_features(
    diagnoses: pd.DataFrame,
    d_icd: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build admission-level diagnosis features keyed by hadm_id."""
    dx = diagnoses.copy()
    dx["icd_code"] = dx["icd_code"].astype(str).str.upper()
    dx["seq_num"] = pd.to_numeric(dx.get("seq_num"), errors="coerce")

    primary = (
        dx.sort_values(["hadm_id", "seq_num"])
        .groupby("hadm_id", observed=True)["icd_code"]
        .first()
        .rename("primary_icd_code")
        .reset_index()
    )

    agg = dx.groupby("hadm_id", observed=True).agg(
        diagnosis_count=("icd_code", "count"),
        unique_diagnosis_count=("icd_code", "nunique"),
    ).reset_index()

    agg = agg.merge(primary, on="hadm_id", how="left")
    agg = agg.merge(build_charlson_flags(dx), on="hadm_id", how="left")

    chronic_patterns = {
        "dx_diabetes": (["E11", "E10", "250"], ["E11", "E10", "250"]),
        "dx_hypertension": (["401", "402"], ["I10"]),
        "dx_ckd": (["585"], ["N18"]),
        "dx_cad": (["414"], ["I25"]),
        "dx_copd_flag": (["491", "492", "496"], ["J44"]),
        "dx_stroke": (["434", "436"], ["I63"]),
    }
    for flag, (icd9, icd10) in chronic_patterns.items():
        mask = _match_mask(dx, icd9, icd10)
        matched = dx.loc[mask, "hadm_id"].drop_duplicates()
        agg[flag] = agg["hadm_id"].isin(matched).astype(np.int8)

    agg["icd_embedding_placeholder"] = ""
    log.info("Diagnosis features: %d admissions × %d features", len(agg), agg.shape[1])
    return agg


def build_comorbidity_pairs(diagnoses: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Co-occurrence matrix for top diagnoses."""
    top_codes = diagnoses["icd_code"].value_counts().head(top_n).index.tolist()
    filtered = diagnoses[diagnoses["icd_code"].isin(top_codes)]

    pairs: Dict[tuple, int] = {}
    for _, group in filtered.groupby("hadm_id", observed=True):
        codes = sorted(set(group["icd_code"].tolist()))
        for i, c1 in enumerate(codes):
            for c2 in codes[i + 1:]:
                pairs[(c1, c2)] = pairs.get((c1, c2), 0) + 1

    records = [{"icd_1": k[0], "icd_2": k[1], "co_occurrence_count": v} for k, v in pairs.items()]
    return pd.DataFrame(records).sort_values("co_occurrence_count", ascending=False)

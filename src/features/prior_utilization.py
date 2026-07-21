"""
src/features/prior_utilization.py
───────────────────────────────────
Engineers prior utilization history (Expansion A) and pre-admission baseline Charlson index (Expansion D)
with strict temporal discipline (lightning fast <1s):
- Prior admissions and diagnoses MUST have dischtime < index admittime.
- First admissions correctly receive 0 prior counts and sentinel days_since_last_discharge = -1.0.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd

from src.features.diagnosis import CHARLSON_WEIGHTS
from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger(__name__)


def build_prior_utilization_features(
    admissions: pd.DataFrame,
    diagnoses: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Lightning-fast (<1s) optimized computation of Expansion A & D features with strict
    dischtime < index admittime temporal discipline.
    """
    log.info("Building Expansion A & D Features (Lightning Fast)...")
    df = admissions[["subject_id", "hadm_id", "admittime", "dischtime"]].copy()
    df["admittime"] = pd.to_datetime(df["admittime"])
    df["dischtime"] = pd.to_datetime(df["dischtime"])

    df["los_days_stay"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400.0
    df["los_days_stay"] = df["los_days_stay"].clip(lower=0.0)

    # 1. Compute Charlson score per hadm_id FIRST if diagnoses provided
    hadm_cci = pd.DataFrame({"hadm_id": df["hadm_id"].unique(), "hadm_cci_score": 0})
    if diagnoses is not None and not diagnoses.empty:
        dx = diagnoses[["hadm_id", "icd_code", "icd_version"]].copy()
        dx["icd_version"] = pd.to_numeric(dx["icd_version"], errors="coerce").fillna(9).astype(int)

        # Normalize ICD codes and slice 3-char prefix ONCE
        clean_codes = dx["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)
        p3 = clean_codes.str.slice(0, 3)
        v9 = dx["icd_version"] == 9
        v10 = dx["icd_version"] == 10

        for condition, mappings in CFG.charlson.items():
            col = f"cci_{condition}"
            icd9_pats = set(p.upper().replace(".", "")[:3] for p in mappings.get("icd9", []))
            icd10_pats = set(p.upper().replace(".", "")[:3] for p in mappings.get("icd10", []))

            mask = (v9 & p3.isin(icd9_pats)) | (v10 & p3.isin(icd10_pats))
            dx[col] = mask.astype(np.int8)

        cci_cols = [f"cci_{c}" for c in CFG.charlson.keys() if f"cci_{c}" in dx.columns]
        if cci_cols:
            dx_hadm = dx.groupby("hadm_id", observed=True)[cci_cols].max().reset_index()
            dx_hadm["hadm_cci_score"] = sum(
                dx_hadm[c] * CHARLSON_WEIGHTS.get(c.replace("cci_", ""), 1) for c in cci_cols
            ).astype(np.int16)
            hadm_cci = dx_hadm[["hadm_id", "hadm_cci_score"]]

    # Merge hadm_cci_score into df
    df = df.merge(hadm_cci, on="hadm_id", how="left")
    df["hadm_cci_score"] = df["hadm_cci_score"].fillna(0).astype(int)

    # 2. Self-join admissions on subject_id to find prior stays
    merged = df.merge(
        df[["subject_id", "hadm_id", "admittime", "dischtime", "los_days_stay", "hadm_cci_score"]],
        on="subject_id",
        suffixes=("_index", "_prior")
    )

    # Strict temporal condition: prior dischtime < index admittime AND different hadm_id
    valid = merged[
        (merged["hadm_id_index"] != merged["hadm_id_prior"]) &
        merged["dischtime_prior"].notna() &
        (merged["dischtime_prior"] < merged["admittime_index"])
    ].copy()

    if valid.empty:
        log.info("No prior admissions found in dataset. Returning sentinel defaults.")
        res = df[["hadm_id"]].copy()
        res["prior_admissions_30d"] = 0
        res["prior_admissions_90d"] = 0
        res["prior_admissions_365d"] = 0
        res["prior_cumulative_los_days"] = 0.0
        res["days_since_last_discharge"] = -1.0
        res["pre_admission_charlson_index"] = 0
        return res

    valid["days_diff"] = (valid["admittime_index"] - valid["dischtime_prior"]).dt.total_seconds() / 86400.0
    valid["is_30d"] = (valid["days_diff"] <= 30.0).astype(int)
    valid["is_90d"] = (valid["days_diff"] <= 90.0).astype(int)
    valid["is_365d"] = (valid["days_diff"] <= 365.0).astype(int)

    # Group by index hadm_id to aggregate prior metrics
    agg = valid.groupby("hadm_id_index", observed=True).agg(
        prior_admissions_30d=("is_30d", "sum"),
        prior_admissions_90d=("is_90d", "sum"),
        prior_admissions_365d=("is_365d", "sum"),
        prior_cumulative_los_days=("los_days_stay_prior", "sum"),
        days_since_last_discharge=("days_diff", "min"),
        pre_admission_charlson_index=("hadm_cci_score_prior", "max"),
    ).reset_index().rename(columns={"hadm_id_index": "hadm_id"})

    res = df[["hadm_id"]].merge(agg, on="hadm_id", how="left")

    res["prior_admissions_30d"] = res["prior_admissions_30d"].fillna(0).astype(int)
    res["prior_admissions_90d"] = res["prior_admissions_90d"].fillna(0).astype(int)
    res["prior_admissions_365d"] = res["prior_admissions_365d"].fillna(0).astype(int)
    res["prior_cumulative_los_days"] = res["prior_cumulative_los_days"].fillna(0.0).astype(float)
    res["days_since_last_discharge"] = res["days_since_last_discharge"].fillna(-1.0).astype(float)
    res["pre_admission_charlson_index"] = res["pre_admission_charlson_index"].fillna(0).astype(int)

    log.info("Expansion A & D Features completed for %d admissions", len(res))
    return res

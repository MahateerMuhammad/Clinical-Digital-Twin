"""
src/features/leakage_filters.py
─────────────────────────────────
Target leakage prevention and feature exclusion lists for Clinical Digital Twin modeling.

Prevents data leakage by removing:
1. Outcome-adjacent columns (e.g. discharge_location, deathtime, dischtime).
2. Diagnosis-derived features finalized after discharge (e.g. charlson_comorbidity_index, cci_*, dx_*).
3. Availability-based leakage (e.g. vitals_*, icu_*, fluids_* when predicting ICU admission).
"""

from __future__ import annotations

import fnmatch
from typing import List, Tuple, Union

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

# Excludes outcome-adjacent columns AND diagnosis-derived features.
# Rationale: ICD codes in diagnoses_icd are often finalized at/after discharge
# and can encode the outcome itself (e.g. a "cardiac arrest" diagnosis code
# in a mortality model is not a predictor, it's the label in disguise).
# A 2025 MIMIC-IV study found readmission models using raw ICD codes hit
# AUROC 0.97+, a strong signature of this exact leakage.
MORTALITY_EXCLUDE = [
    # Direct outcome and duration leakage
    "deathtime", "dischtime", "discharge_location", "los_days", "los_hours", "dod",
    # Diagnosis-derived post-hoc ICD leakage
    "charlson_comorbidity_index", "cci_*", "dx_*", "primary_icd_code", "icd_embedding_placeholder",
    # Readmission deterministic proxies (patients who die cannot be readmitted)
    "readmission_30d", "next_admittime", "days_to_readmission", "readmit_*",
    # Post-hoc ICU stay accumulation metrics
    "icu_los_days", "n_icu_stays", "has_icu_stay", "icu_*",
]

# Run C: Strict 24h Early Observation Window (Excludes all full-stay aggregates, last lab values, slopes, care-unit transfers, and clinical notes)
MORTALITY_EXCLUDE_RUN_C = MORTALITY_EXCLUDE + [
    # Full-admission count and duration aggregates (observation window leakage)
    "medication_count", "unique_medications", "med_duration_hours_mean", "med_duration_hours_max",
    "unique_diagnosis_count", "unique_procedure_count", "major_procedure_count", "has_major_procedure",
    # Full-admission lab trajectory metrics (last/slope/change/std/count)
    "lab_*_last", "lab_*_slope", "lab_*_change", "lab_*_std", "lab_*_count", "lab_*_abnormal_count", "lab_*_missing_ratio",
    # Care unit & transfer indicators
    "first_careunit", "last_careunit", "intime", "outtime",
    # Clinical notes and text readability features
    "note_type", "charttime", "text_clean", "readability_flesch", "text_tfidf_ready",
]

READMISSION_EXCLUDE = [
    "next_admittime", "days_to_readmission", "deathtime", "dischtime", "discharge_location",
]

# Strict 24h Early Observation Window Readmission Filter (Living Cohort + 24h Window Discipline)
READMISSION_EXCLUDE_STRICT = [
    # Post-hoc resolution and timing proxies
    "next_admittime", "days_to_readmission", "readmit_*", "deathtime", "dischtime", "discharge_location", "dod", "los_days", "los_hours", "hospital_expire_flag",
    # Post-hoc ICD diagnosis & comorbidity exclusions
    "charlson_comorbidity_index", "cci_*", "dx_*", "primary_icd_code", "icd_embedding_placeholder",
    # Strict 24-hour observation window exclusions (full-stay counts, duration aggregates, lab trajectories, care units, text)
    "medication_count", "unique_medications", "med_duration_hours_mean", "med_duration_hours_max",
    "unique_diagnosis_count", "unique_procedure_count", "major_procedure_count", "has_major_procedure",
    "lab_*_last", "lab_*_slope", "lab_*_change", "lab_*_std", "lab_*_count", "lab_*_abnormal_count", "lab_*_missing_ratio",
    "first_careunit", "last_careunit", "intime", "outtime",
    "note_type", "charttime", "text_clean", "readability_flesch", "text_tfidf_ready",
    # Post-hoc ICU stay accumulation metrics
    "icu_los_days", "n_icu_stays", "has_icu_stay", "icu_*",
]

# icu_* / fluids_* / vitals_* features are only populated for admissions that
# already had an ICU stay — using them to predict ICU admission is leaking
# the label through feature-availability itself (non-null pattern == the answer).
ICU_ADMISSION_EXCLUDE = [
    "icu_los_days", "n_icu_stays", "has_icu_stay",
    "icu_*", "fluids_*", "vitals_*",
    "first_careunit", "last_careunit",
]

# Strict 24h Early Observation Window / Admission-Time ICU Risk prediction filter
# Excludes all post-admission aggregates, slopes, lasts, notes readability, and care unit transfers.
ICU_ADMISSION_EXCLUDE_STRICT = ICU_ADMISSION_EXCLUDE + [
    # Post-hoc ICD diagnosis & comorbidity exclusions (leakage from current stay)
    "charlson_comorbidity_index", "cci_*", "dx_*",
    # Full-admission count and duration aggregates (observation window leakage)
    "medication_count", "unique_medications", "med_duration_hours_mean", "med_duration_hours_max",
    "unique_diagnosis_count", "unique_procedure_count", "major_procedure_count", "has_major_procedure",
    "med_class_*",
    # Full-admission lab trajectory metrics (last/slope/change/std/count)
    "lab_*_last", "lab_*_slope", "lab_*_change", "lab_*_std", "lab_*_count", "lab_*_abnormal_count", "lab_*_missing_ratio",
    "lab_unique_items",
    "lab_*_median", "lab_*_min", "lab_*_max", "lab_*_wb_count", "lab_*_wb_missing_ratio",
    "lab_*_wb_abnormal_count", "lab_*_poc_abnormal_count", "lab_*_poc_missing_ratio",
    # Care unit & transfer indicators
    "intime", "outtime",
    # Clinical notes and text readability features
    "note_type", "charttime", "text_clean", "readability_flesch", "text_tfidf_ready",
]

LOS_EXCLUDE = [
    "dischtime", "discharge_location", "deathtime",
]

# Strict 24h Early Observation Window / Admission-Time Length of Stay filter
# Excludes direct target proxies, post-hoc ICD codes, full-stay aggregates, notes readability, and care unit transfers.
LOS_EXCLUDE_STRICT = LOS_EXCLUDE + [
    # Direct target/outcome & resolution proxies
    "deathtime", "dischtime", "discharge_location", "los_days", "los_hours", "dod", "hospital_expire_flag",
    "next_admittime", "days_to_readmission", "readmission_30d", "readmit_*",
    "icu_los_days", "n_icu_stays", "has_icu_stay", "icu_*", "fluids_*", "vitals_*",
    # Post-hoc ICD diagnosis & comorbidity exclusions (leakage from current stay)
    "charlson_comorbidity_index", "cci_*", "dx_*", "primary_icd_code", "icd_embedding_placeholder",
    # Full-admission count and duration aggregates (observation window leakage)
    "medication_count", "unique_medications", "med_duration_hours_mean", "med_duration_hours_max",
    "unique_diagnosis_count", "unique_procedure_count", "major_procedure_count", "has_major_procedure",
    "med_class_*",
    # Full-admission lab trajectory metrics (last/slope/change/std/count)
    "lab_*_last", "lab_*_slope", "lab_*_change", "lab_*_std", "lab_*_count", "lab_*_abnormal_count", "lab_*_missing_ratio",
    "lab_unique_items",
    "lab_*_median", "lab_*_min", "lab_*_max", "lab_*_wb_count", "lab_*_wb_missing_ratio",
    "lab_*_wb_abnormal_count", "lab_*_poc_abnormal_count", "lab_*_poc_missing_ratio",
    # Care unit & transfer indicators
    "first_careunit", "last_careunit", "intime", "outtime",
    # Clinical notes and text readability features
    "note_type", "charttime", "text_clean", "readability_flesch", "text_tfidf_ready",
]

DETERIORATION_EXCLUDE = [
    "dischtime", "deathtime", "discharge_location", "dod", "los_days", "los_hours",
    "hospital_expire_flag", "first_careunit*", "last_careunit*", "intime*", "outtime*",
    "icu_los_days", "n_icu_stays", "has_icu_stay", "next_admittime", "days_to_readmission", "readmission_30d",
]

# Strict Clinical Deterioration Exclusion List
# Excludes direct target proxies, post-hoc care unit transfers, discharge outcomes, post-hoc ICD coding,
# ICU chartevents vitals (availability leakage), full-stay accumulation metrics, and full-stay lab order frequency/median features.
DETERIORATION_EXCLUDE_STRICT = DETERIORATION_EXCLUDE + [
    # Post-hoc ICD diagnosis & comorbidity exclusions (leakage from current stay resolution)
    "charlson_comorbidity_index", "cci_*", "dx_*", "primary_icd_code", "icd_embedding_placeholder",
    # ICU chartevents vitals (100% missing for ward target=0, 0% missing for target=1 -> availability leakage)
    "vital_*", "news2_*",
    # Full-admission accumulation counts & duration metrics (observation window leakage)
    "medication_count", "unique_medications", "med_duration_hours_mean", "med_duration_hours_max",
    "unique_diagnosis_count", "unique_procedure_count", "major_procedure_count", "has_major_procedure",
    "unique_*", "*_count", "*_abnormal_count", "*_missing_ratio", "lab_unique_items", "*_ratio",
    # Full-admission lab trajectory medians, extremes, lasts (window overflow leakage)
    "lab_*_median", "lab_*_last", "lab_*_max", "lab_*_min", "lab_*_std", "lab_*_slope", "lab_*_change",
    # Clinical notes and text readability features
    "note_type", "charttime", "text_clean", "readability_flesch", "text_tfidf_ready",
]


def match_column_patterns(columns: List[str], patterns: List[str]) -> List[str]:
    """
    Match DataFrame columns against a list of exact column names and wildcard patterns (e.g., 'prefix_*').

    Parameters
    ----------
    columns : list[str]
        List of DataFrame column names.
    patterns : list[str]
        List of exact column names or wildcard patterns (e.g. 'cci_*', 'vitals_*').

    Returns
    -------
    list[str]
        List of matching column names present in columns.
    """
    matched = set()
    for pattern in patterns:
        if "*" in pattern or "?" in pattern or "[" in pattern:
            matches = fnmatch.filter(columns, pattern)
            matched.update(matches)
        else:
            if pattern in columns:
                matched.add(pattern)
    return sorted(list(matched))


def apply_exclusions(
    df: pd.DataFrame,
    pattern_list: List[str],
    verbose: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[str]]]:
    """
    Remove columns matching exclusion patterns from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    pattern_list : list[str]
        List of exact column names or wildcard patterns to exclude.
    verbose : bool
        Whether to log before/after column counts and dropped columns.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with leakage columns removed.
    """
    cols_to_drop = match_column_patterns(list(df.columns), pattern_list)
    filtered_df = df.drop(columns=cols_to_drop)

    if verbose:
        log.info(
            "apply_exclusions: %d → %d columns (dropped %d columns)",
            len(df.columns), len(filtered_df.columns), len(cols_to_drop),
        )
        if cols_to_drop:
            log.info("Dropped columns (%d): %s", len(cols_to_drop), cols_to_drop[:15])
            if len(cols_to_drop) > 15:
                log.info("  ... and %d more columns", len(cols_to_drop) - 15)

    return filtered_df


def check_availability_leakage(
    df: pd.DataFrame,
    feature_prefix: str,
    target_col: str,
) -> pd.DataFrame:
    """
    Diagnostic check for feature-availability leakage.

    Prints and returns percentage of rows per target class that have non-null
    values in columns matching feature_prefix.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    feature_prefix : str
        Prefix or pattern matching feature columns (e.g., 'icu_*', 'vitals_*').
    target_col : str
        Name of target column (e.g., 'has_icu_stay', 'hospital_expire_flag').

    Returns
    -------
    pd.DataFrame
        Summary table with columns [target_class, n_rows, n_with_feature, pct_with_feature].
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    pattern = feature_prefix if "*" in feature_prefix else f"{feature_prefix}*"
    matched_cols = match_column_patterns(list(df.columns), [pattern])

    if not matched_cols:
        log.warning("No columns matched pattern '%s' in DataFrame.", feature_prefix)
        return pd.DataFrame()

    has_feature_data = df[matched_cols].notna().any(axis=1)

    results = []
    print(f"\n============================================================")
    print(f" Availability Leakage Diagnostic: {feature_prefix} vs {target_col}")
    print(f" Matched columns ({len(matched_cols)}): {matched_cols[:5]}...")
    print(f"============================================================")

    for target_val, group in df.groupby(target_col, observed=True):
        group_mask = has_feature_data.loc[group.index]
        n_rows = len(group)
        n_has_feat = int(group_mask.sum())
        pct_has_feat = (n_has_feat / n_rows * 100.0) if n_rows > 0 else 0.0

        print(f" Target '{target_col}' = {target_val}:")
        print(f"   Total rows           : {n_rows:,}")
        print(f"   Rows with {feature_prefix:10s} : {n_has_feat:,} ({pct_has_feat:.2f}%)")

        results.append({
            "target_col": target_col,
            "target_value": target_val,
            "n_rows": n_rows,
            "n_with_feature": n_has_feat,
            "pct_with_feature": round(pct_has_feat, 2),
        })

    print(f"============================================================\n")
    return pd.DataFrame(results)

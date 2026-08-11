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
# Numeric statistics derived from discharge summaries.
#
# The discharge summary is authored *at discharge*, so its length, sentence count,
# medical-keyword count and negation count describe the entire admission. They are
# not observable within a 24-hour window and act as a proxy for stay duration and
# acuity.
#
# The strict lists previously excluded only the text columns themselves
# (`text_clean`, `text_tfidf_ready`, `readability_flesch`, `note_type`) and let
# these numeric derivatives through. In the Phase 1 Run C model they ranked 4th
# (`sentence_count`, mean |SHAP| 0.337) and 9th (`medical_keyword_count`, 0.181)
# of 66 features — materially inflating a metric published as "strict 24h".
DISCHARGE_NOTE_DERIVED = [
    "sentence_count", "word_count", "char_count",
    "medical_keyword_count", "negation_count",
    "note_count", "note_id", "note_*",
]

# Count of distinct lab assays ordered across the whole admission: a full-stay
# ordering-intensity proxy, not a 24-hour observation.
FULL_STAY_LAB_COUNTERS = ["lab_unique_items", "lab_total_count"]

# Drug-class flags from src/features/medication.py. These are computed as
# ``groupby("hadm_id").max()`` over the *entire* prescriptions table with no time
# filter, so they mean "prescribed at any point during the admission" — the same
# full-stay aggregation that already disqualifies `medication_count` and
# `unique_medications` from the strict lists.
#
# `med_class_opioid` is the clearest offender: 94.8% of admissions ending in death
# carry the flag versus 56.6% of survivors (1.68x). That gradient is comfort-care
# prescribing — morphine given *because* care is being withdrawn — recorded days
# after the 24-hour window closes. `med_class_statin` shows no such gradient
# (0.93x), confirming the effect is specific to end-of-life drug classes rather
# than a general "sicker patients receive more drugs" association.
#
# In Phase 1 Run C it ranked 1st of 62 features (mean |SHAP| 1.035 XGBoost,
# 0.722 LightGBM), 1.6x the next feature.
FULL_STAY_MED_CLASS = ["med_class_*"]

# Whole-admission lab aggregates. src/features/laboratory.py uses `charttime` only
# to *order* records, never to filter them, so `lab_wbc_max` is the maximum over
# the entire stay and `lab_bicarbonate_min` for a patient dying of acidosis on day
# 12 is the value measured as they die. Presenting these as a 24-hour observation
# is the same defect as the note statistics, but larger: they were 30 of the 40
# lab features surviving Run C, and `lab_bicarbonate_min` ranked 2nd by SHAP.
#
# `lab_*_first` is excluded here too, not because it leaks — 92.2% of first draws
# fall inside 24h — but because `lab_*_first_24h` is the same value computed
# inside an explicit window, so keeping both would merely duplicate the column.
#
# The strict protocols consume WINDOWED_LAB_FEATURES instead; see
# :func:`src.features.laboratory.build_lab_features_windowed`.
FULL_STAY_LAB_AGGREGATES = [
    "lab_*_mean", "lab_*_min", "lab_*_max", "lab_*_median", "lab_*_first",
]
# ``lab_*_mean`` was missing from this list until 2026-08-01. It went unnoticed
# because feature selection happened to drop the whole-stay mean columns for
# unrelated reasons; once selection was corrected to preserve windowed labs, nine
# of them survived and entered Run C, where `lab_wbc_mean`, `lab_bun_mean`,
# `lab_glucose_mean`, `lab_bicarbonate_mean` and `lab_platelets_mean` immediately
# appeared in the top ten SHAP features. A mean over the whole admission includes
# values measured after the 24h window and up to the moment of death.
#
# The lesson is that this list must enumerate aggregate suffixes exhaustively
# rather than by memory. The full vocabulary produced by the lab builder is:
# count, mean, median, min, max, std, missing_ratio, abnormal_count, first, last,
# slope, change. Every one is either excluded here or in MORTALITY_EXCLUDE_RUN_C.

# The 24h-windowed counterparts. Full-stay protocols (mortality Run A/B,
# readmission Run A) exclude these so their feature sets stay exactly as they were
# before the window was introduced, keeping their metrics comparable with the
# pre-correction baseline snapshots.
WINDOWED_LAB_FEATURES = ["lab_*_24h"]


# ── Emergency-department families ─────────────────────────────────────────────
#
# ED features are the first ward-grade physiology in this pipeline, and they are
# genuinely pre-admission (99.7% of ED intime precedes admittime). But the ED
# module covers only 37.1% of the cohort, so *presence* carries information that
# has nothing to do with the patient's state, and that is the failure mode which
# forced the Phase 5 rebuild. These lists exist so the risk is named and
# switchable rather than buried in a missingness pattern.

# Never a feature under any protocol. `disposition` is the ED outcome; `outtime`
# is the departure timestamp whose modelling-safe derivative is `ed_los_hours`.
ED_OUTCOME_DERIVED = ["disposition", "ed_disposition", "ed_outtime"]

# The explicit "this patient came through the ED" flag. Excluded from primary
# models: it is a cohort-membership indicator, not a measurement, and a tree will
# happily split on it to recover the ED subpopulation's base rate.
ED_AVAILABILITY_FEATURES = ["ed_available"]

# A nurse's 1-5 ESI severity judgement. It partly *causes* the ICU decision
# rather than predicting it — the same shape as the testing-volume features
# removed in the Phase 5 rebuild — so it belongs in a sensitivity arm and is
# excluded from every primary model.
ED_ACUITY_FEATURES = ["ed_triage_acuity"]

# How often the patient was observed, not what was observed. A patient recorded
# 14 times in the ED was being watched closely; that is a clinician's concern
# leaking in as a feature, exactly like the ICU `vital_*_count` family.
ED_MONITORING_INTENSITY = ["ed_vital_*_count", "ed_n_stays"]

# Everything an ED-free protocol must drop to stay comparable with the
# pre-ED baseline snapshots.
ED_ALL_FEATURES = ["ed_*"]

# The default guard for a primary model: keep the physiology, drop the artefacts.
ED_EXCLUDE_PRIMARY = (
    ED_OUTCOME_DERIVED + ED_AVAILABILITY_FEATURES
    + ED_ACUITY_FEATURES + ED_MONITORING_INTENSITY
)


MORTALITY_EXCLUDE = [
    # Direct outcome and duration leakage
    "deathtime", "dischtime", "discharge_location", "los_days", "los_hours", "dod",
    # Diagnosis-derived post-hoc ICD leakage
    "charlson_comorbidity_index", "cci_*", "dx_*", "primary_icd_code", "icd_embedding_placeholder",
    # Readmission deterministic proxies (patients who die cannot be readmitted)
    "readmission_30d", "next_admittime", "days_to_readmission", "readmit_*",
    # Post-hoc ICU stay accumulation metrics
    "icu_los_days", "n_icu_stays", "has_icu_stay", "icu_*",
] + ED_EXCLUDE_PRIMARY

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
] + DISCHARGE_NOTE_DERIVED + FULL_STAY_LAB_COUNTERS + FULL_STAY_MED_CLASS + FULL_STAY_LAB_AGGREGATES

# Run B (full-stay, leak-free). Kept separate from MORTALITY_EXCLUDE because
# MORTALITY_EXCLUDE_RUN_C inherits that list and must *keep* the windowed columns.
MORTALITY_EXCLUDE_RUN_B = MORTALITY_EXCLUDE + WINDOWED_LAB_FEATURES

READMISSION_EXCLUDE = [
    "next_admittime", "days_to_readmission", "deathtime", "dischtime", "discharge_location",
] + WINDOWED_LAB_FEATURES + ED_EXCLUDE_PRIMARY

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
] + DISCHARGE_NOTE_DERIVED + FULL_STAY_LAB_COUNTERS + FULL_STAY_MED_CLASS + FULL_STAY_LAB_AGGREGATES + ED_EXCLUDE_PRIMARY

# icu_* / fluids_* / vitals_* features are only populated for admissions that
# already had an ICU stay — using them to predict ICU admission is leaking
# the label through feature-availability itself (non-null pattern == the answer).
ICU_ADMISSION_EXCLUDE = [
    "icu_los_days", "n_icu_stays", "has_icu_stay",
    "icu_*", "fluids_*", "vitals_*",
    "first_careunit", "last_careunit",
] + ED_EXCLUDE_PRIMARY

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
] + DISCHARGE_NOTE_DERIVED + FULL_STAY_LAB_COUNTERS + FULL_STAY_LAB_AGGREGATES

LOS_EXCLUDE = [
    "dischtime", "discharge_location", "deathtime",
] + ED_EXCLUDE_PRIMARY

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
] + DISCHARGE_NOTE_DERIVED + FULL_STAY_LAB_COUNTERS + FULL_STAY_LAB_AGGREGATES

DETERIORATION_EXCLUDE = [
    "dischtime", "deathtime", "discharge_location", "dod", "los_days", "los_hours",
    "hospital_expire_flag", "first_careunit*", "last_careunit*", "intime*", "outtime*",
    "icu_los_days", "n_icu_stays", "has_icu_stay", "next_admittime", "days_to_readmission", "readmission_30d",
] + ED_EXCLUDE_PRIMARY

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
] + DISCHARGE_NOTE_DERIVED + FULL_STAY_LAB_COUNTERS + FULL_STAY_MED_CLASS + FULL_STAY_LAB_AGGREGATES


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

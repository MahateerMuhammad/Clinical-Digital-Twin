import time
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config import CFG
from src.data.cleaner import DataCleaner
from src.data.merger import TableMerger
from src.features.vitals import build_vital_features_from_df
from src.features.laboratory import build_lab_features_from_df
from src.features.medication import build_medication_features
from src.features.feature_selection import prepare_features
from src.features.dataset_builder import DatasetBuilder
from src.utils.logger import get_logger

log = get_logger("harsh_test")

def run_harsh_stress_test():
    print("=" * 70)
    print("  CLINICAL DIGITAL TWIN — HARSH STRESS & ADVERSARIAL EDGE-CASE TEST")
    print("=" * 70)
    start_time = time.time()
    cleaner = DataCleaner()
    merger = TableMerger()

    # -------------------------------------------------------------------------
    # TEST 1: Adversarial & Corrupted String/Date Inputs
    # -------------------------------------------------------------------------
    print("\n[STRESS TEST 1] Testing Corrupted Timestamps, Malformed Whitespace & Categoricals...")
    adv_patients = pd.DataFrame({
        "subject_id": [1001, 1002, 1003, 1004],
        "gender": ["  M ", "F  ", "  ", None],
        "anchor_age": [45, -5, 120, 300],  # negative & out of range ages
        "dod": ["2026-01-01 10:00:00", "INVALID_DATE_STRING", None, "2026-12-31"]
    })
    
    cleaned_pat, r_pat = cleaner.clean_table(adv_patients, "patients", id_cols=["subject_id"], save=False)
    assert cleaned_pat["gender"].tolist() == ["M", "F", np.nan, np.nan], f"Gender cleanup failed: {cleaned_pat['gender'].tolist()}"
    assert pd.isna(cleaned_pat.loc[1, "dod"]), "Invalid date string was not coerced to NaT!"
    print("  ✓ Corrupted datetimes coerced to NaT without crashing.")
    print("  ✓ Categorical whitespace & empty strings normalized to NaN.")

    # -------------------------------------------------------------------------
    # TEST 2: Inverted Timestamps & Out-of-Range Clinical Outliers
    # -------------------------------------------------------------------------
    print("\n[STRESS TEST 2] Testing Reversed Timestamps & Extreme Clinical Outliers...")
    adv_inputevents = pd.DataFrame({
        "subject_id": [1001, 1001, 1002],
        "hadm_id": [2001, 2001, 2002],
        "stay_id": [3001, 3001, 3002],
        "starttime": ["2026-01-02 12:00:00", "2026-01-01 10:00:00", "2026-01-01 10:00:00"],
        "endtime":   ["2026-01-01 10:00:00", "2026-01-01 12:00:00", "2026-01-01 12:00:00"],  # Row 0 inverted
        "amount": [50.0, 999999999.0, 100.0],  # Row 1 impossible amount
        "patientweight": [70.0, 0.2, 900.0],   # Row 1 & 2 impossible weights
        "rate": [10.0, 50.0, 2000000.0]        # Row 2 impossible rate
    })
    
    cleaned_inp, r_inp = cleaner.clean_table(adv_inputevents, "inputevents", save=False)
    assert cleaned_inp.loc[0, "_invalid_time_order"] == 1, "Reversed starttime/endtime was not flagged!"
    assert pd.isna(cleaned_inp.loc[1, "amount"]), "Impossible input amount (999M) was not capped to NaN!"
    assert pd.isna(cleaned_inp.loc[1, "patientweight"]), "Impossible patient weight (0.2kg) was not capped to NaN!"
    assert pd.isna(cleaned_inp.loc[2, "patientweight"]), "Impossible patient weight (900kg) was not capped to NaN!"
    print("  ✓ Timestamp inversions correctly flagged with _invalid_time_order=1.")
    print("  ✓ Impossible clinical outliers (amount > 10M, weight < 1kg or > 500kg) capped to NaN.")

    # -------------------------------------------------------------------------
    # TEST 3: Fan-Out & Zero-Stay Merge Stress
    # -------------------------------------------------------------------------
    print("\n[STRESS TEST 3] Testing 1-to-Many Table Merges & Zero-Stay Fan-Out Prevention...")
    admissions = pd.DataFrame({
        "subject_id": [1001, 1002, 1003],
        "hadm_id": [2001, 2002, 2003],
        "admittime": pd.to_datetime(["2026-01-01", "2026-01-05", "2026-01-10"]),
        "dischtime": pd.to_datetime(["2026-01-04", "2026-01-08", "2026-01-15"]),
        "hospital_expire_flag": [0, 1, 0]
    })
    
    icustays = pd.DataFrame({
        "subject_id": [1001, 1001, 1001],  # Hadm 2001 has 3 ICU stays! Hadm 2002/2003 have 0 ICU stays
        "hadm_id": [2001, 2001, 2001],
        "stay_id": [3001, 3002, 3003],
        "intime": pd.to_datetime(["2026-01-01 10:00:00", "2026-01-02 10:00:00", "2026-01-03 10:00:00"]),
        "outtime": pd.to_datetime(["2026-01-01 22:00:00", "2026-01-02 22:00:00", "2026-01-03 22:00:00"]),
        "los": [0.5, 0.5, 0.5]
    })

    merged_adm = merger.merge_admission_level(admissions, adv_patients)
    assert len(merged_adm) == len(admissions), f"Merge fan-out detected! Expected {len(admissions)} rows, got {len(merged_adm)}"
    print("  ✓ Admission-level merge preserved exact row count (zero fan-out).")

    # -------------------------------------------------------------------------
    # TEST 4: Empty & Single-Row Edge Cases in Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[STRESS TEST 4] Testing Empty DataFrames & Single-Row Edge Cases in Feature Builders...")
    empty_meds = build_medication_features(pd.DataFrame())
    assert "hadm_id" in empty_meds.columns and len(empty_meds) == 0, "Empty medication dataframe failed gracefully."
    
    single_lab = pd.DataFrame({
        "hadm_id": [2001],
        "itemid": [50912],  # Creatinine
        "valuenum": [1.2],
        "charttime": ["2026-01-01 10:00:00"],
        "_is_duplicate": [0]
    })
    lab_feats = build_lab_features_from_df(single_lab)
    assert len(lab_feats) == 1, "Single-row lab dataframe feature building failed!"
    assert lab_feats.iloc[0]["lab_creatinine_mean"] == 1.2, "Single-row lab mean calculation incorrect!"
    print("  ✓ Feature builders handle empty & single-row inputs cleanly without errors.")

    # -------------------------------------------------------------------------
    # TEST 5: Feature Selection Thresholds & Inf/NaN Filtering
    # -------------------------------------------------------------------------
    print("\n[STRESS TEST 5] Testing Feature Selection Thresholds & Zero-Variance Elimination...")
    feat_df = pd.DataFrame({
        "hadm_id": [2001, 2002, 2003, 2004],
        "hospital_expire_flag": [0, 1, 0, 0],
        "zero_var_feat": [1.0, 1.0, 1.0, 1.0],         # Constant feature -> must be dropped
        "all_nan_feat": [np.nan, np.nan, np.nan, np.nan], # 100% missing -> must be dropped
        "good_feat": [1.5, 8.2, 0.1, 9.4],             # High variance non-linear -> keep
        "corr_feat1": [1.0, 2.0, 3.0, 4.0],
        "corr_feat2": [1.0001, 2.0001, 3.0001, 4.0001] # 0.9999 correlation with corr_feat1 -> 1 must be dropped
    })
    
    selected_df, report = prepare_features(feat_df, id_cols=["hadm_id"], target_cols=["hospital_expire_flag"])
    assert "zero_var_feat" not in selected_df.columns, "Zero-variance feature was not dropped!"
    assert "all_nan_feat" not in selected_df.columns, "All-NaN feature was not dropped!"
    assert "good_feat" in selected_df.columns, "High-quality feature was erroneously dropped!"
    assert not ("corr_feat1" in selected_df.columns and "corr_feat2" in selected_df.columns), "Collinear features (>0.95 correlation) were not deduplicated!"
    print("  ✓ Feature selection dropped zero-variance, 100% NaN, and highly collinear features.")

    # -------------------------------------------------------------------------
    # TEST 6: Full End-to-End Dataset Construction on Stress Fixtures
    # -------------------------------------------------------------------------
    print("\n[STRESS TEST 6] End-to-End Dataset Building on Stress Fixture Cohort...")
    dataset_builder = DatasetBuilder()
    
    mock_cleaned = {
        "patients": adv_patients,
        "admissions": admissions,
        "icustays": icustays,
        "diagnoses_icd": pd.DataFrame({"subject_id": [1001], "hadm_id": [2001], "icd_code": ["I10"], "seq_num": [1], "_is_duplicate": [0]}),
        "procedures_icd": pd.DataFrame(columns=["subject_id", "hadm_id", "icd_code", "seq_num", "_is_duplicate"]),
        "labevents": single_lab,
        "prescriptions": pd.DataFrame({"subject_id": [1001], "hadm_id": [2001], "pharmacy_id": [1], "drug": ["Aspirin"], "starttime": ["2026-01-01"], "stoptime": ["2026-01-02"], "_is_duplicate": [0]}),
        "pharmacy": pd.DataFrame(),
        "emar": pd.DataFrame(),
        "emar_detail": pd.DataFrame(),
        "chartevents": pd.DataFrame({"subject_id": [1001], "hadm_id": [2001], "stay_id": [3001], "itemid": [220045], "valuenum": [80.0], "_is_duplicate": [0]}),
        "inputevents": adv_inputevents,
        "outputevents": pd.DataFrame(),
        "radiology_detail": pd.DataFrame(),
        "discharge": pd.DataFrame()
    }
    
    mock_features = {
        "laboratory": lab_feats,
        "medication": build_medication_features(mock_cleaned["prescriptions"]),
        "icu_stay": pd.DataFrame({"stay_id": [3001, 3002, 3003], "hadm_id": [2001, 2001, 2001], "icu_los": [0.5, 0.5, 0.5]}),
        "fluids": pd.DataFrame()
    }
    
    constructed = dataset_builder.build_all(mock_cleaned, mock_features)
    adm_dataset = constructed.get("admission_level", pd.DataFrame())
    
    assert len(adm_dataset) == 3, f"Expected 3 admissions in final dataset, got {len(adm_dataset)}"
    assert adm_dataset["hospital_expire_flag"].isna().sum() == 0, "Target column contains null values!"
    print(f"  ✓ Constructed admission-level dataset ({len(adm_dataset)} rows × {adm_dataset.shape[1]} cols) with 0 null targets.")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  ALL HARSH STRESS & ADVERSARIAL TESTS PASSED SUCCESSFULLY ({total_time:.2f} s)")
    print("=" * 70)

if __name__ == "__main__":
    run_harsh_stress_test()

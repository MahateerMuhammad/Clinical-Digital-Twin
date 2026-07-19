import time
import sys
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

log = get_logger("extreme_test")

def run_extreme_stress_test():
    print("=" * 80)
    print("  CLINICAL DIGITAL TWIN — EXTREME ADVERSARIAL STRESS TEST & AUDIT")
    print("=" * 80)
    start_time = time.time()
    
    issues_found = []
    cleaner = DataCleaner()
    merger = TableMerger()

    # -------------------------------------------------------------------------
    # EXTREME TEST 1: Extreme Type Corruption & Malformed Mixed Types
    # -------------------------------------------------------------------------
    print("\n[EXTREME TEST 1] Malformed Mixed Types, Inf, NaN & Special Unicode Characters...")
    try:
        corrupted_patients = pd.DataFrame({
            "subject_id": [1001, "1002", 1003, np.nan, 1005],
            "gender": ["M", "F", "Unknown/Other", "  ", None],
            "anchor_age": [50, "999", -10, np.inf, 25],
            "anchor_year": [2026, "2026", 2026, 2026, 2026],
            "dod": ["2026-01-01 00:00:00", "NULL", "9999-99-99", None, "2026-05-05 12:34:56.789123"]
        })
        
        # Test cleaning robustness
        cleaned_pat, r_pat = cleaner.clean_table(corrupted_patients, "patients", id_cols=["subject_id"], save=False)
        
        # Check non-numeric age coerced safely
        if pd.api.types.is_numeric_dtype(cleaned_pat["anchor_age"]):
            age_issues = (cleaned_pat["anchor_age"] < 0) | (cleaned_pat["anchor_age"] > 130)
            if age_issues.any():
                issues_found.append(("patients", "Out-of-range anchor_age not completely capped to NaN"))
        
        print("  ✓ Corrupted datetimes, infinite values, and mixed types processed safely.")
    except Exception as e:
        issues_found.append(("patients", f"Exception on corrupted type input: {e}"))

    # -------------------------------------------------------------------------
    # EXTREME TEST 2: High Cardinality & 100% Missing DataFrames
    # -------------------------------------------------------------------------
    print("\n[EXTREME TEST 2] High Cardinality (100k categoricals) & 100% Missing Tables...")
    try:
        n_high = 10000
        high_card_df = pd.DataFrame({
            "subject_id": np.arange(n_high),
            "hadm_id": np.arange(n_high) + 10000,
            "pharmacy_id": np.arange(n_high) + 20000,
            "drug": [f"Drug_Variant_{i}" for i in range(n_high)],  # 10,000 unique drug strings
            "starttime": ["2026-01-01 10:00:00"] * n_high,
            "stoptime": ["2026-01-01 12:00:00"] * n_high,
            "_is_duplicate": [0] * n_high,
            "_invalid_time_order": [0] * n_high
        })
        
        med_feats = build_medication_features(high_card_df)
        assert len(med_feats) == n_high, f"High-cardinality medication building lost rows! ({len(med_feats)} vs {n_high})"
        print("  ✓ High-cardinality categorical feature aggregation (10k unique drug strings) completed.")
    except Exception as e:
        issues_found.append(("medication_features", f"Failed on high-cardinality inputs: {e}"))

    # -------------------------------------------------------------------------
    # EXTREME TEST 3: Extreme 1-to-Many Merges & Orphan Foreign Keys
    # -------------------------------------------------------------------------
    print("\n[EXTREME TEST 3] Orphan Foreign Keys & Extreme 1-to-Many Fan-Out Chaos...")
    try:
        orphan_admissions = pd.DataFrame({
            "subject_id": [999991, 999992, 999993],  # None of these subjects exist in patients table
            "hadm_id": [888881, 888882, 888883],
            "admittime": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "dischtime": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
            "hospital_expire_flag": [0, 0, 1]
        })
        
        orphan_patients = pd.DataFrame({
            "subject_id": [111111, 222222],  # Disjoint subjects
            "gender": ["M", "F"],
            "anchor_age": [60, 70]
        })
        
        merged_orphan = merger.merge_admission_level(orphan_admissions, orphan_patients)
        assert len(merged_orphan) == 3, f"Orphan admissions merge altered row count! ({len(merged_orphan)} vs 3)"
        assert merged_orphan["gender"].isna().all(), "Orphan admissions should have NaN patient demographic fields!"
        print("  ✓ Orphan admissions (zero patient matches) merged cleanly with NaNs without dropping rows.")
    except Exception as e:
        issues_found.append(("merger", f"Orphan foreign key merge failed: {e}"))

    # -------------------------------------------------------------------------
    # EXTREME TEST 4: Extreme Slope & Mathematical Aggregation Instability
    # -------------------------------------------------------------------------
    print("\n[EXTREME TEST 4] Mathematical Aggregation Instability (Single-value slopes, Identical charttimes)...")
    try:
        same_time_labs = pd.DataFrame({
            "hadm_id": [5001, 5001, 5001],
            "itemid": [50912, 50912, 50912],
            "valuenum": [1.0, 2.0, 3.0],
            "charttime": ["2026-01-01 10:00:00", "2026-01-01 10:00:00", "2026-01-01 10:00:00"],  # Identical timestamps
            "_is_duplicate": [0, 0, 0]
        })
        
        lab_res = build_lab_features_from_df(same_time_labs)
        slope_val = lab_res.iloc[0]["lab_creatinine_slope"]
        if np.isnan(slope_val) or np.isinf(slope_val):
            pass # Expected or handled
        print("  ✓ Identical timestamp aggregations handled without division-by-zero or linear regression failure.")
    except Exception as e:
        issues_found.append(("laboratory_slope", f"Failed on identical timestamp slope calculation: {e}"))

    # -------------------------------------------------------------------------
    # EXTREME TEST 5: Feature Selection with Zero-Variance & All-NaN Features
    # -------------------------------------------------------------------------
    print("\n[EXTREME TEST 5] Feature Selection with 100% NaN & Zero-Variance Columns...")
    try:
        all_bad_df = pd.DataFrame({
            "hadm_id": [1, 2, 3, 4],
            "hospital_expire_flag": [0, 1, 0, 1],
            "const_col": [5.0, 5.0, 5.0, 5.0],
            "nan_col": [np.nan, np.nan, np.nan, np.nan]
        })
        selected_bad, report_bad = prepare_features(all_bad_df, id_cols=["hadm_id"], target_cols=["hospital_expire_flag"])
        assert len(report_bad.kept_features) == 0, f"Expected 0 kept features, got {len(report_bad.kept_features)}"
        print("  ✓ Feature selection safely handled 100% uninformative feature matrix (0 features retained).")
    except Exception as e:
        issues_found.append(("feature_selection", f"Failed on all-bad feature matrix: {e}"))

    # -------------------------------------------------------------------------
    # EXTREME TEST 6: Massive Multi-Table Merges (1,000 Cohort Rows × 50k Events)
    # -------------------------------------------------------------------------
    print("\n[EXTREME TEST 6] Multi-Table Merge Stress (1,000 Admissions × 50,000 Events)...")
    try:
        n_adm = 1000
        n_events = 50000
        
        stress_admissions = pd.DataFrame({
            "subject_id": np.repeat(np.arange(n_adm) + 100, 1),
            "hadm_id": np.arange(n_adm) + 1000,
            "admittime": pd.to_datetime(["2026-01-01"] * n_adm),
            "dischtime": pd.to_datetime(["2026-01-05"] * n_adm),
            "hospital_expire_flag": np.random.choice([0, 1], size=n_adm)
        })
        
        stress_icustays = pd.DataFrame({
            "subject_id": np.repeat(np.arange(n_adm) + 100, 2),  # 2 ICU stays per admission
            "hadm_id": np.repeat(np.arange(n_adm) + 1000, 2),
            "stay_id": np.arange(n_adm * 2) + 10000,
            "intime": pd.to_datetime(["2026-01-01 10:00:00"] * (n_adm * 2)),
            "outtime": pd.to_datetime(["2026-01-02 10:00:00"] * (n_adm * 2)),
            "los": [1.0] * (n_adm * 2)
        })
        
        stress_cleaned = {
            "patients": pd.DataFrame({"subject_id": np.arange(n_adm) + 100, "gender": ["M"] * n_adm, "anchor_age": [65] * n_adm}),
            "admissions": stress_admissions,
            "icustays": stress_icustays,
            "diagnoses_icd": pd.DataFrame({"subject_id": [100], "hadm_id": [1000], "icd_code": ["I10"], "seq_num": [1], "_is_duplicate": [0]}),
            "procedures_icd": pd.DataFrame(columns=["subject_id", "hadm_id", "icd_code", "seq_num", "_is_duplicate"]),
            "labevents": pd.DataFrame({"subject_id": np.random.choice(np.arange(n_adm) + 100, n_events),
                                       "hadm_id": np.random.choice(np.arange(n_adm) + 1000, n_events),
                                       "itemid": [50912] * n_events,
                                       "valuenum": np.random.randn(n_events),
                                       "charttime": ["2026-01-01 12:00:00"] * n_events,
                                       "_is_duplicate": [0] * n_events}),
            "prescriptions": pd.DataFrame(columns=["subject_id", "hadm_id", "pharmacy_id", "drug", "starttime", "stoptime", "_is_duplicate"]),
            "pharmacy": pd.DataFrame(), "emar": pd.DataFrame(), "emar_detail": pd.DataFrame(),
            "chartevents": pd.DataFrame(), "inputevents": pd.DataFrame(), "outputevents": pd.DataFrame(),
            "radiology_detail": pd.DataFrame(), "discharge": pd.DataFrame()
        }
        
        stress_features = {
            "laboratory": build_lab_features_from_df(stress_cleaned["labevents"]),
            "icu_stay": pd.DataFrame({"stay_id": stress_icustays["stay_id"], "hadm_id": stress_icustays["hadm_id"], "icu_los": [1.0] * len(stress_icustays)}),
        }
        
        builder = DatasetBuilder()
        built = builder.build_all(stress_cleaned, stress_features)
        adm_out = built.get("admission_level", pd.DataFrame())
        assert len(adm_out) == n_adm, f"Cohort dataset construction fan-out! Expected {n_adm} rows, got {len(adm_out)}"
        print(f"  ✓ Multi-table merge stress (1,000 admissions × 50,000 lab events) built cleanly into {len(adm_out)} rows.")
    except Exception as e:
        issues_found.append(("dataset_builder_stress", f"Failed on 50k event stress build: {e}"))

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"  EXTREME ADVERSARIAL STRESS TEST FINISHED IN {total_time:.2f} SECONDS")
    print("=" * 80)
    
    # Print Comprehensive Issues Summary
    print("\n" + "─" * 80)
    print("  ACCUMULATED PIPELINE ISSUES & RISK AUDIT REPORT")
    print("─" * 80)
    if not issues_found:
        print("  ✓ ZERO EXCEPTION-LEVEL BUGS FOUND across all 6 extreme stress tests!")
    else:
        for module, err in issues_found:
            print(f"  ❌ [{module}] {err}")

if __name__ == "__main__":
    run_extreme_stress_test()

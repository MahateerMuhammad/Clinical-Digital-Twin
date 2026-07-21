"""
verify_phase0.py
────────────────
Verification script for Phase 0 — Shared Prerequisites.
"""

from pathlib import Path
import pandas as pd

from src.data.splits import create_patient_splits
from src.features.leakage_filters import (
    MORTALITY_EXCLUDE,
    READMISSION_EXCLUDE,
    ICU_ADMISSION_EXCLUDE,
    LOS_EXCLUDE,
    apply_exclusions,
    check_availability_leakage,
)
from src.utils.logger import get_logger

log = get_logger("verify_phase0")

def verify_phase0():
    print("\n" + "=" * 60)
    print("      PHASE 0 VERIFICATION: SHARED PREREQUISITES")
    print("=" * 60 + "\n")

    # 1. Test Split Generation & Assertions
    print("[STEP 1] Generating and Verifying Patient-Level Split...")
    split_df = create_patient_splits()
    split_file = Path("data/processed/patient_split.parquet")

    assert split_file.exists(), "Error: patient_split.parquet was not created!"
    
    # Reload split file from disk to verify disk output
    disk_split = pd.read_parquet(split_file)
    print(f"  ✓ Saved file found: {split_file} ({split_file.stat().st_size / 1024:.1f} KB)")
    print(f"  ✓ Rows in split file: {len(disk_split):,}")

    train_pts = set(disk_split[disk_split["split"] == "train"]["subject_id"])
    val_pts = set(disk_split[disk_split["split"] == "val"]["subject_id"])
    test_pts = set(disk_split[disk_split["split"] == "test"]["subject_id"])

    # Zero overlap check
    assert len(train_pts & val_pts) == 0, "FAILED: train & val overlap!"
    assert len(train_pts & test_pts) == 0, "FAILED: train & test overlap!"
    assert len(val_pts & test_pts) == 0, "FAILED: val & test overlap!"
    
    print("  ✓ PASSED ASSERTION: Zero subject_id overlap across train/val/test splits!")
    print(f"    - Train patients: {len(train_pts):,}")
    print(f"    - Val patients  : {len(val_pts):,}")
    print(f"    - Test patients : {len(test_pts):,}")

    # 2. Test Leakage Exclusion Lists on admission_level_selected.parquet
    print("\n[STEP 2] Testing Leakage Exclusions on admission_level_selected.parquet...")
    adm_file = Path("data/processed/admission_level_selected.parquet")
    assert adm_file.exists(), f"Error: {adm_file} not found!"
    
    df_adm = pd.read_parquet(adm_file)
    orig_cols = len(df_adm.columns)
    print(f"  Input Dataset: {len(df_adm):,} rows × {orig_cols} columns\n")

    # Test Mortality Exclusion
    df_mort = apply_exclusions(df_adm, MORTALITY_EXCLUDE, verbose=False)
    dropped_mort = set(df_adm.columns) - set(df_mort.columns)
    print(f"  1. Mortality Exclusions:")
    print(f"     Before: {orig_cols} cols -> After: {len(df_mort.columns)} cols (Dropped {len(dropped_mort)} columns)")
    print(f"     Dropped: {sorted(list(dropped_mort))[:10]}...")

    # Test Readmission Exclusion
    df_readm = apply_exclusions(df_adm, READMISSION_EXCLUDE, verbose=False)
    dropped_readm = set(df_adm.columns) - set(df_readm.columns)
    print(f"\n  2. Readmission Exclusions:")
    print(f"     Before: {orig_cols} cols -> After: {len(df_readm.columns)} cols (Dropped {len(dropped_readm)} columns)")
    print(f"     Dropped: {sorted(list(dropped_readm))}")

    # Test ICU Admission Exclusion
    df_icu = apply_exclusions(df_adm, ICU_ADMISSION_EXCLUDE, verbose=False)
    dropped_icu = set(df_adm.columns) - set(df_icu.columns)
    print(f"\n  3. ICU Admission Exclusions:")
    print(f"     Before: {orig_cols} cols -> After: {len(df_icu.columns)} cols (Dropped {len(dropped_icu)} columns)")
    print(f"     Dropped: {sorted(list(dropped_icu))[:10]}...")

    # Test LOS Exclusion
    df_los = apply_exclusions(df_adm, LOS_EXCLUDE, verbose=False)
    dropped_los = set(df_adm.columns) - set(df_los.columns)
    print(f"\n  4. Length of Stay (LOS) Exclusions:")
    print(f"     Before: {orig_cols} cols -> After: {len(df_los.columns)} cols (Dropped {len(dropped_los)} columns)")
    print(f"     Dropped: {sorted(list(dropped_los))}")

    # 3. Test Availability Leakage Diagnostic
    print("\n[STEP 3] Running Diagnostic Check for Feature-Availability Leakage...")
    check_availability_leakage(df_adm, "icu_*", "has_icu_stay")
    check_availability_leakage(df_adm, "vitals_*", "has_icu_stay")

    print("=" * 60)
    print("  PHASE 0 VERIFICATION COMPLETE: ALL REQUIREMENTS SATISFIED!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    verify_phase0()

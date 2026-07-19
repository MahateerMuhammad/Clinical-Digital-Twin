import time
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.config import CFG
from src.data.cleaner import DataCleaner
from src.features.build_features import FeatureBuilder
from src.features.dataset_builder import DatasetBuilder
from src.utils.logger import get_logger

log = get_logger("smoke_test")

def run_smoke_test():
    print("=" * 60)
    print("  CLINICAL DIGITAL TWIN — FULL PIPELINE SMOKE TEST")
    print("=" * 60)
    start_time = time.time()
    
    interim_dir = CFG.resolve(CFG.paths.interim)
    tables = {}
    
    table_names = [
        "patients", "admissions", "icustays", "diagnoses_icd", "procedures_icd",
        "labevents", "prescriptions", "pharmacy", "emar", "emar_detail",
        "chartevents", "inputevents", "outputevents", "radiology_detail", "discharge"
    ]
    
    print("\n[STEP 1] Loading interim cleaned parquet datasets...")
    for name in table_names:
        p_path = interim_dir / f"{name}_clean.parquet"
        if p_path.exists():
            if name in ["emar_detail", "chartevents", "labevents", "emar", "pharmacy", "discharge", "inputevents", "prescriptions", "outputevents"]:
                cols = ["_is_duplicate"]
                if name == "prescriptions":
                    cols = ["subject_id", "hadm_id", "pharmacy_id", "drug", "starttime", "stoptime", "_is_duplicate"]
                elif name == "chartevents":
                    cols = ["subject_id", "hadm_id", "stay_id", "itemid", "valuenum", "_is_duplicate"]
                elif name == "labevents":
                    cols = ["subject_id", "hadm_id", "itemid", "valuenum", "_is_duplicate"]
                elif name == "emar":
                    cols = ["subject_id", "hadm_id", "medication", "_is_duplicate"]
                elif name == "pharmacy":
                    cols = ["subject_id", "hadm_id", "pharmacy_id", "medication", "_is_duplicate"]
                elif name == "discharge":
                    cols = ["subject_id", "hadm_id", "_is_duplicate"]
                elif name == "inputevents":
                    cols = ["subject_id", "hadm_id", "stay_id", "amount", "_is_duplicate"]
                elif name == "outputevents":
                    cols = ["subject_id", "hadm_id", "stay_id", "value", "_is_duplicate"]
                df = pd.read_parquet(p_path, columns=cols)
            else:
                df = pd.read_parquet(p_path)
            tables[name] = df
            print(f"  ✓ {name:20s}: {len(df):>10,} rows × {df.shape[1]:>2} cols ({p_path.stat().st_size / (1024*1024):>6.1f} MB on disk)")
        else:
            print(f"  ⚠ {name:20s}: NOT FOUND at {p_path}")

    # 2. Check Data Cleaning Metrics
    print("\n[STEP 2] Auditing Data Cleaning & Duplicate Integrity...")
    cleaner = DataCleaner()
    dup_summary = {}
    for name, df in tables.items():
        if "_is_duplicate" in df.columns:
            n_dup = int((df["_is_duplicate"] == 1).sum())
            pct = 100.0 * n_dup / max(len(df), 1)
            dup_summary[name] = (n_dup, pct)
            print(f"  ✓ {name:20s}: {n_dup:>8,} duplicate rows ({pct:>5.2f}%)")
            
            # Assertions for reasonable duplicate bounds
            if name == "prescriptions":
                assert pct < 0.10, f"Prescriptions duplicate rate unexpectedly high ({pct:.2f}%)"
            elif name == "chartevents":
                assert pct < 0.01, f"Chartevents duplicate rate unexpectedly high ({pct:.2f}%)"
            elif name == "radiology_detail":
                if len(df) > 100:
                    assert pct < 0.05, f"Radiology detail duplicate rate unexpectedly high ({pct:.2f}%)"

    # 3. Feature Engineering Audit
    print("\n[STEP 3] Executing Feature Engineering Modules...")
    fb_start = time.time()
    feature_builder = FeatureBuilder()
    
    # We sample a subset of admissions if full feature engineering on 60M chartevents is heavy
    sample_admissions = set(tables["admissions"]["hadm_id"].sample(n=min(5000, len(tables["admissions"])), random_state=42))
    
    sampled_tables = {}
    for name, df in tables.items():
        if "hadm_id" in df.columns:
            sampled_tables[name] = df[df["hadm_id"].isin(sample_admissions)].copy()
        elif "subject_id" in df.columns:
            sample_subjects = set(tables["admissions"].loc[tables["admissions"]["hadm_id"].isin(sample_admissions), "subject_id"])
            sampled_tables[name] = df[df["subject_id"].isin(sample_subjects)].copy()
        else:
            sampled_tables[name] = df.copy()
            
    features = feature_builder.build_all(sampled_tables, use_chunked=False)
    fb_time = time.time() - fb_start
    print(f"  ✓ Feature engineering completed in {fb_time:.2f} seconds.")
    for feat_name, feat_df in features.items():
        print(f"    - {feat_name:20s}: {len(feat_df):>6,} rows × {feat_df.shape[1]:>3} cols")
        # Check no infinite values or broken columns
        num_cols = feat_df.select_dtypes(include=[np.number]).columns
        inf_mask = np.isinf(feat_df[num_cols].values).any()
        assert not inf_mask, f"Infinite values detected in feature set '{feat_name}'!"

    # 4. Dataset Building Audit
    print("\n[STEP 4] Constructing Admission-Level & ICU Stay-Level Datasets...")
    db_start = time.time()
    dataset_builder = DatasetBuilder()
    datasets = dataset_builder.build_all(sampled_tables, features)
    db_time = time.time() - db_start
    
    adm_ds = datasets.get("admission_level", pd.DataFrame())
    print(f"  ✓ Dataset creation completed in {db_time:.2f} seconds.")
    print(f"    - Admission-Level Dataset: {len(adm_ds):,} rows × {adm_ds.shape[1]:,} cols")
    
    # Validate row count matches admission sample exactly
    expected_rows = len(sampled_tables["admissions"])
    assert len(adm_ds) == expected_rows, f"Row count mismatch in admission dataset! ({len(adm_ds)} vs {expected_rows})"
    
    # Validate target columns exist and have non-null targets
    target_col = CFG.targets.mortality_inhosp
    assert target_col in adm_ds.columns, f"Target column '{target_col}' missing from admission dataset!"
    n_null_target = adm_ds[target_col].isna().sum()
    assert n_null_target == 0, f"Detected {n_null_target} null targets in mortality_inhosp!"

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  ALL SMOKE TEST CHECKS PASSED IN {total_time:.2f} SECONDS!")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()

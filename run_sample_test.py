import time
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.config import CFG
from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.merger import TableMerger
from src.features.build_features import FeatureBuilder
from src.features.dataset_builder import DatasetBuilder

def run_sample_benchmark():
    print("=== STEP 3: 5,000 REAL SUBJECT SAMPLE BENCHMARK ===")
    start_time = time.time()
    
    # 1. Load first 5,000 subjects from patients.csv
    patients_raw = pd.read_csv("data/raw/hosp/patients.csv")
    sample_subjects = set(patients_raw["subject_id"].unique()[:5000])
    print(f"Sampled {len(sample_subjects)} real subjects.")
    
    def sample_processor(chunk: pd.DataFrame) -> pd.DataFrame:
        if "subject_id" in chunk.columns:
            return chunk[chunk["subject_id"].isin(sample_subjects)]
        return chunk

    loader = DataLoader(validate=False)
    cleaner = DataCleaner()
    feature_builder = FeatureBuilder()
    dataset_builder = DatasetBuilder()
    
    table_names = ["patients", "admissions", "icustays", "diagnoses_icd", "procedures_icd", 
                   "labevents", "prescriptions", "pharmacy", "emar", "emar_detail", 
                   "chartevents", "inputevents", "outputevents", "radiology_detail", "discharge"]
    
    tables = {}
    for name in table_names:
        file_path = CFG.table_file(name)
        if file_path.exists():
            df, _ = loader._load(name, max_chunks=None, row_processor=sample_processor)
            df = sample_processor(df)
            tables[name] = df
            print(f"  Sample '{name}': {len(df):,} rows")
    
    load_time = time.time() - start_time
    print(f"\n[TIMING] Sample load: {load_time:.2f} seconds ({load_time/60:.2f} min)")
    
    # 2. Cleaning
    clean_start = time.time()
    cleaned = {}
    dup_rates = {}
    for name, df in tables.items():
        t_cfg = CFG.tables.get(name, {})
        cleaned_df, report = cleaner.clean_table(df, name, id_cols=t_cfg.get("id_cols"))
        cleaned[name] = cleaned_df
        if "_is_duplicate" in cleaned_df.columns and len(cleaned_df) > 0:
            dup_rates[name] = float(cleaned_df["_is_duplicate"].mean())
            
    clean_time = time.time() - clean_start
    print(f"[TIMING] Sample clean: {clean_time:.2f} seconds")
    print("Sample duplicate rates:")
    for name, rate in dup_rates.items():
        print(f"  - {name}: {rate:.2%}")
        
    # Check duplicate rates in plausible ranges
    assert dup_rates.get("chartevents", 0) < 0.05, "Chartevents duplicate rate too high!"
    assert dup_rates.get("radiology_detail", 0) < 0.01, "Radiology detail duplicate rate too high!"
    assert dup_rates.get("emar_detail", 0) < 0.05, "EMAR detail duplicate rate too high!"
    assert dup_rates.get("prescriptions", 0) < 0.55, "Prescriptions duplicate rate unexpected!"

    # 3. Feature building
    feat_start = time.time()
    features = feature_builder.build_all(cleaned, use_chunked=False)
    feat_time = time.time() - feat_start
    print(f"[TIMING] Sample feature engineering: {feat_time:.2f} seconds")
    
    # Check lab missing_ratio
    if "laboratory" in features and not features["laboratory"].empty:
        lab_feats = features["laboratory"]
        assert "lab_creatinine_missing_ratio" in lab_feats.columns
        assert not lab_feats["lab_creatinine_missing_ratio"].isna().any(), "Unexpected NaNs in lab_missing_ratio!"
        
    # 4. Dataset creation
    ds_start = time.time()
    datasets = dataset_builder.build_all(cleaned, features)
    admission_ds = datasets.get("admission_level", pd.DataFrame())
    ds_time = time.time() - ds_start
    print(f"[TIMING] Sample dataset creation: {ds_time:.2f} seconds")
    
    total_time = time.time() - start_time
    
    # Assert admission level row count equals sample admission count
    n_sample_admissions = len(cleaned["admissions"])
    n_ds_admissions = len(admission_ds)
    print(f"\nAdmission dataset rows: {n_ds_admissions:,} (expected {n_sample_admissions:,})")
    assert n_ds_admissions == n_sample_admissions, f"Admission dataset row count mismatch! ({n_ds_admissions} vs {n_sample_admissions})"
    
    print(f"\n==================================================")
    print(f"TOTAL 5,000-SUBJECT BENCHMARK RUNTIME: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"==================================================")

if __name__ == "__main__":
    run_sample_benchmark()

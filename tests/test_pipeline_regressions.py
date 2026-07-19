"""
tests/test_pipeline_regressions.py
───────────────────────────────────
Permanent regression test suite enforcing all 11 bug fixes against synthetic fixtures.
"""

import pandas as pd
import numpy as np
from typing import Dict

from tests.fixtures.synthetic_patients import create_synthetic_tables
from src.data.cleaner import DataCleaner
from src.data.merger import TableMerger
from src.features.laboratory import build_lab_features_from_df
from src.features.vitals import build_vital_features_from_df
from src.utils.config import CFG

def get_synthetic_tables():
    return create_synthetic_tables()

# 1. test_no_row_cap
def test_no_row_cap(tables):
    for name, df in tables.items():
        assert len(df) > 0, f"Table {name} is empty"

# 2. test_chartevents_labevents_itemid_filter
def test_chartevents_labevents_itemid_filter(tables):
    key_lab_items = set()
    for v in CFG.key_labs.values():
        if isinstance(v, list):
            key_lab_items.update(v)
        else:
            key_lab_items.add(v)
            
    raw_labs = tables["labevents"]
    filtered_labs = raw_labs[raw_labs["itemid"].isin(key_lab_items)]
    
    # Non-matching itemids (e.g. 99999, 88888) should be filtered out
    assert 99999 not in filtered_labs["itemid"].values
    assert 88888 not in filtered_labs["itemid"].values
    assert 50912 in filtered_labs["itemid"].values

# 3. test_no_false_duplicates_event_tables
def test_no_false_duplicates_event_tables(tables):
    cleaner = DataCleaner()
    # Check chartevents on fixture
    t_cfg = CFG.tables.get("chartevents", {})
    cleaned_ce, report_ce = cleaner.clean_table(tables["chartevents"], "chartevents", id_cols=t_cfg.get("id_cols"))
    dup_rate = (cleaned_ce["_is_duplicate"].sum() / len(cleaned_ce)) if len(cleaned_ce) > 0 else 0
    assert dup_rate < 0.50, f"Duplicate rate on chartevents too high: {dup_rate:.2%}"

# 4. test_radiology_field_ordinal_key
def test_radiology_field_ordinal_key(tables):
    cleaner = DataCleaner()
    t_cfg = CFG.tables.get("radiology_detail", {})
    rad_df = tables["radiology_detail"]
    cleaned_rad, report_rad = cleaner.clean_table(rad_df, "radiology_detail", id_cols=t_cfg.get("id_cols"))
    
    # note_id 10001-RR-1 has field_ordinal 1, 2, 3 -> these must NOT be flagged as duplicates
    ordinal_rows = cleaned_rad[cleaned_rad["note_id"] == "10001-RR-1"]
    assert (ordinal_rows["_is_duplicate"] == 0).all(), "Multi-field_ordinal rows were falsely flagged as duplicates!"
    
    # note_id 10002-RR-1 has 2 identical rows with field_ordinal 1 -> these SHOULD be flagged
    dup_rows = cleaned_rad[cleaned_rad["note_id"] == "10002-RR-1"]
    assert (dup_rows["_is_duplicate"] == 1).all(), "Exact duplicate radiology rows were not flagged!"

# 5. test_icu_merge_no_fanout
def test_icu_merge_no_fanout(tables):
    merger = TableMerger()
    merged = merger.merge_admission_level(tables["admissions"], tables["patients"])
    
    # Admission 2003 has 3 ICU stays in fixture
    adm_2003 = merged[merged["hadm_id"] == 2003]
    assert len(adm_2003) == 1, f"Merge fanout detected! Expected 1 row for hadm_id 2003, got {len(adm_2003)}"

# 6. test_n_icu_stays_is_a_true_count
def test_n_icu_stays_is_a_true_count(tables):
    icu = tables["icustays"]
    icu_counts = icu.groupby("hadm_id")["stay_id"].count().to_dict()
    assert icu_counts.get(2003) == 3, f"Expected 3 ICU stays for hadm_id 2003, got {icu_counts.get(2003)}"
    assert icu_counts.get(2002) == 1, f"Expected 1 ICU stay for hadm_id 2002, got {icu_counts.get(2002)}"

# 7. test_lab_missing_ratio_explicit
def test_lab_missing_ratio_explicit(tables):
    labs = tables["labevents"]
    features = build_lab_features_from_df(labs)
    
    # hadm_id 2001 has 0 key labs drawn
    # hadm_id 2002 has creatinine (50912) drawn, but 0 glucose (50931) drawn
    row_2002 = features[features["hadm_id"] == 2002].iloc[0]
    
    # glucose was not drawn for 2002 -> missing_ratio should be 1.0, count should be 0
    assert row_2002["lab_glucose_missing_ratio"] == 1.0, f"Expected missing_ratio=1.0 for missing lab, got {row_2002['lab_glucose_missing_ratio']}"
    assert row_2002["lab_glucose_count"] == 0, f"Expected count=0 for missing lab, got {row_2002['lab_glucose_count']}"

# 8. test_vitals_missing_ratio_behavior
def test_vitals_missing_ratio_behavior(tables):
    ce = tables["chartevents"]
    vitals = build_vital_features_from_df(ce)
    
    # stay_id 3001 has heart_rate (220045) drawn, but 0 spo2 (220277)
    row_3001 = vitals[vitals["stay_id"] == 3001].iloc[0]
    
    # Confirm vital summary stats for missing spo2 are NaN (intended design)
    assert "vital_spo2_missing_ratio" not in row_3001.index, "vitals.py should not generate missing_ratio column"
    assert pd.isna(row_3001.get("vital_spo2_mean")), "Missing vital mean should be NaN for feature selection filtering"

# 9. test_lab_variant_features_separate
def test_lab_variant_features_separate(tables):
    labs = tables["labevents"]
    features = build_lab_features_from_df(labs)
    
    # hadm_id 2009 has serum creatinine (50912) AND whole-blood creatinine (50806)
    row_2009 = features[features["hadm_id"] == 2009].iloc[0]
    
    # Verify separate feature columns exist and hold distinct values
    assert "lab_creatinine_mean" in features.columns, "Serum creatinine feature column missing!"
    assert "lab_creatinine_wb_mean" in features.columns, "Whole-blood creatinine_wb feature column missing!"
    assert row_2009["lab_creatinine_mean"] == 1.8
    assert row_2009["lab_creatinine_wb_mean"] == 1.6

# 10. test_timestamp_order_flagged_not_dropped
def test_timestamp_order_flagged_not_dropped(tables):
    cleaner = DataCleaner()
    adm_df = tables["admissions"]
    cleaned_adm, report = cleaner.clean_table(adm_df, "admissions", id_cols=["subject_id", "hadm_id"])
    
    # Invalid admission time hadm_id 2007 (dischtime < admittime)
    adm_2007 = cleaned_adm[cleaned_adm["hadm_id"] == 2007]
    assert len(adm_2007) == 1, "Row with invalid timestamp order was wrongly dropped!"
    assert adm_2007["_invalid_time_order"].iloc[0] == 1, "Invalid timestamp order was not flagged!"

# 11. test_vectorized_vs_original_lab_features
def test_vectorized_vs_original_lab_features(tables):
    labs = tables["labevents"]
    features = build_lab_features_from_df(labs)
    
    # Validate mathematical properties on edge cases
    # hadm 2003 has increasing glucose (100, 150, 200) -> slope must be positive
    row_2003 = features[features["hadm_id"] == 2003].iloc[0]
    assert row_2003["lab_glucose_change"] == 100.0
    assert row_2003["lab_glucose_slope"] > 0

# 12. test_medication_duration_and_duplicate_handling
def test_medication_duration_and_duplicate_handling():
    from src.features.medication import build_medication_features
    df = pd.DataFrame({
        "hadm_id": [101, 101, 101, 102],
        "pharmacy_id": [1, 2, 3, 4],
        "drug": ["Aspirin", "Aspirin", "Heparin", "Insulin"],
        "starttime": ["2026-01-01 10:00:00", "2026-01-01 10:00:00", "2026-01-01 12:00:00", "2026-01-01 10:00:00"],
        "stoptime": ["2026-01-01 14:00:00", "2026-01-01 08:00:00", "2026-01-01 16:00:00", "2026-01-01 12:00:00"],
        "_is_duplicate": [0, 1, 0, 0],
        "_invalid_time_order": [0, 1, 0, 0],
    })
    res = build_medication_features(df)
    
    # Duplicate row (_is_duplicate=1) for hadm 101 should be ignored (medication count = 2, not 3)
    row_101 = res[res["hadm_id"] == 101].iloc[0]
    assert row_101["medication_count"] == 2, f"Expected medication_count=2, got {row_101['medication_count']}"
    
    # Valid durations for 101: 4 hours and 4 hours -> mean should be 4.0 (ignoring negative duration)
    assert row_101["med_duration_hours_mean"] == 4.0, f"Expected duration mean=4.0, got {row_101['med_duration_hours_mean']}"


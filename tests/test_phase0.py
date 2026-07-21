"""
tests/test_phase0.py
────────────────────
Unit tests for Phase 0 — Shared Prerequisites (splits and leakage filters).
"""

import unittest
import numpy as np
import pandas as pd

from src.data.splits import create_patient_splits
from src.features.leakage_filters import (
    MORTALITY_EXCLUDE,
    READMISSION_EXCLUDE,
    ICU_ADMISSION_EXCLUDE,
    LOS_EXCLUDE,
    apply_exclusions,
    check_availability_leakage,
    match_column_patterns,
)


class TestPhase0(unittest.TestCase):

    def test_match_column_patterns(self):
        columns = [
            "subject_id", "hadm_id", "cci_copd", "cci_diabetes", "dx_123",
            "vitals_hr_mean", "icu_los_days", "has_icu_stay", "deathtime"
        ]
        
        # Exact match
        matched_exact = match_column_patterns(columns, ["deathtime", "subject_id"])
        self.assertEqual(set(matched_exact), {"deathtime", "subject_id"})

        # Wildcard prefix match
        matched_prefix = match_column_patterns(columns, ["cci_*", "vitals_*"])
        self.assertEqual(set(matched_prefix), {"cci_copd", "cci_diabetes", "vitals_hr_mean"})

    def test_apply_exclusions(self):
        df = pd.DataFrame({
            "subject_id": [1, 2],
            "hadm_id": [10, 20],
            "deathtime": ["2020-01-01", None],
            "dischtime": ["2020-01-02", "2020-01-03"],
            "cci_copd": [1, 0],
            "cci_dementia": [0, 0],
            "age": [65, 70],
        })

        filtered = apply_exclusions(df, MORTALITY_EXCLUDE, verbose=False)

        # Excluded columns should be removed
        self.assertNotIn("deathtime", filtered.columns)
        self.assertNotIn("dischtime", filtered.columns)
        self.assertNotIn("cci_copd", filtered.columns)
        self.assertNotIn("cci_dementia", filtered.columns)

        # Unrelated columns must remain intact
        self.assertIn("subject_id", filtered.columns)
        self.assertIn("hadm_id", filtered.columns)
        self.assertIn("age", filtered.columns)

    def test_create_patient_splits(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mock_patients = pd.DataFrame({"subject_id": list(range(100))})
            mock_file = tmp_path / "mock_patient_level.parquet"
            mock_patients.to_parquet(mock_file)

            out_file = tmp_path / "patient_split.parquet"
            split_df = create_patient_splits(
                admission_path=mock_file,
                output_path=out_file,
                seed=42,
                ratios=(0.70, 0.15, 0.15),
            )

            self.assertTrue(out_file.exists())
            self.assertEqual(len(split_df), 100)

            train_pts = set(split_df[split_df["split"] == "train"]["subject_id"])
            val_pts = set(split_df[split_df["split"] == "val"]["subject_id"])
            test_pts = set(split_df[split_df["split"] == "test"]["subject_id"])

            # Zero overlap check
            self.assertEqual(len(train_pts & val_pts), 0)
            self.assertEqual(len(train_pts & test_pts), 0)
            self.assertEqual(len(val_pts & test_pts), 0)
            self.assertEqual(len(train_pts) + len(val_pts) + len(test_pts), 100)

    def test_check_availability_leakage(self):
        df = pd.DataFrame({
            "has_icu_stay": [0, 0, 1, 1],
            "icu_los_days": [np.nan, np.nan, 2.5, 3.1],
            "age": [50, 60, 70, 80],
        })

        res = check_availability_leakage(df, "icu_*", "has_icu_stay")
        self.assertFalse(res.empty)
        self.assertEqual(len(res), 2)


if __name__ == "__main__":
    unittest.main()

"""
src/features/dataset_builder.py
─────────────────────────────────
Create final ML-ready datasets at multiple granularity levels.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.data.merger import TableMerger
from src.features.interactions import build_interaction_features
from src.features.feature_selection import (
    FeatureSelectionReport,
    generate_correlation_report,
    generate_missing_report,
    prepare_features,
)
from src.utils.config import CFG
from src.utils.io_utils import save_parquet
from src.utils.reports import df_to_markdown
from src.utils.logger import get_logger

log = get_logger(__name__)


class DatasetBuilder:
    """Produce patient, admission, ICU, time-series, notes, and similarity datasets."""

    def __init__(self) -> None:
        self.cfg = CFG
        self.merger = TableMerger()
        self.processed_dir = Path(self.cfg.resolve(self.cfg.paths.processed))

    def build_all(
        self,
        tables: Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        apply_feature_selection: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Build and save all target datasets."""
        admissions = tables.get("admissions", pd.DataFrame())
        patients = tables.get("patients", pd.DataFrame())
        icustays = tables.get("icustays", pd.DataFrame())

        if admissions.empty:
            raise ValueError("admissions table required for dataset creation")

        admissions = self.merger.add_icu_flags(admissions, icustays)

        hadm_features = {
            k: v for k, v in features.items()
            if v is not None and not v.empty and "hadm_id" in v.columns
        }

        admission_df = self.merger.merge_admission_level(admissions, patients, hadm_features)
        # Interaction terms (age×diabetes, creatinine×age, cci×age, …) — must run
        # after the feature merge so the source columns exist. Previously these were
        # only produced by an unused helper, so they never reached the datasets.
        admission_df = build_interaction_features(admission_df)
        patient_df = self.merger.merge_patient_level(admission_df)

        icu_features = {}
        if "vitals" in features:
            icu_features["vitals"] = features["vitals"]
        if "fluids" in features:
            icu_features["fluids"] = features["fluids"]
        icu_df = self.merger.merge_icu_level(icustays, admissions, icu_features)

        ts_df = self.merger.build_time_series(
            admissions,
            labs=tables.get("labevents"),
            vitals=features.get("vitals"),
            meds=tables.get("prescriptions"),
        )

        notes_df = features.get("notes", pd.DataFrame())
        similarity_df = self.merger.build_similarity_dataset(
            admission_df, features.get("diagnosis"),
        )

        datasets = {
            "patient_level": patient_df,
            "admission_level": admission_df,
            "icu_level": icu_df,
            "time_series": ts_df,
            "clinical_notes": notes_df,
            "similarity": similarity_df,
        }

        fs_report: Optional[FeatureSelectionReport] = None
        if apply_feature_selection and not admission_df.empty:
            selected, fs_report = prepare_features(admission_df)
            datasets["admission_level_selected"] = selected
            self._save_feature_selection_reports(selected, fs_report)

        self._save_datasets(datasets)
        self._generate_documentation(datasets, fs_report)
        return datasets

    def _save_datasets(self, datasets: Dict[str, pd.DataFrame]) -> None:
        for name, df in datasets.items():
            if df is not None and not df.empty:
                path = self.processed_dir / f"{name}.parquet"
                save_parquet(df, path)
                log.info("Saved dataset '%s' → %s (%d rows)", name, path.name, len(df))

    def _save_feature_selection_reports(
        self,
        df: pd.DataFrame,
        report: FeatureSelectionReport,
    ) -> None:
        tables_dir = Path(self.cfg.resolve(self.cfg.paths.tables))
        tables_dir.mkdir(parents=True, exist_ok=True)

        save_parquet(generate_missing_report(df), tables_dir / "feature_missing_report.parquet")
        generate_correlation_report(df, str(tables_dir / "feature_correlation_report.parquet"))

        with open(self.cfg.resolve("reports/feature_engineering_report.md"), "w", encoding="utf-8") as fh:
            fh.write("# Feature Engineering Report\n\n")
            fh.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
            fh.write("## Feature Selection Summary\n\n")
            for k, v in report.to_dict().items():
                if k not in ["dropped_features", "kept_features", "highly_correlated_pairs"]:
                    fh.write(f"- **{k}**: {v}\n")
            
            fh.write("\n## Dropped Features (Categorized by Reason)\n\n")
            fh.write("### Genuine Clinical Missingness (> Threshold)\n")
            fh.write("*(Note: In prior runs, all vitals/labs were dropped here due to the 1.5M row-cap bug. These are now true clinical drops.)*\n")
            for feat in sorted(report.high_missing)[:50]:
                fh.write(f"- {feat}\n")
            if len(report.high_missing) > 50: fh.write(f"- _... and {len(report.high_missing)-50} more_\n")
            
            fh.write("\n### Constant / Zero Variance\n")
            for feat in sorted(report.constant_features + report.near_zero_variance)[:50]:
                fh.write(f"- {feat}\n")
            if len(report.constant_features + report.near_zero_variance) > 50: fh.write("- _... and more_\n")
            
            fh.write("\n### Duplicates / Highly Correlated\n")
            corr_drops = [p[1] for p in report.highly_correlated_pairs]
            for feat in sorted(report.duplicate_features + corr_drops)[:50]:
                fh.write(f"- {feat}\n")
            if len(report.duplicate_features + corr_drops) > 50: fh.write("- _... and more_\n")

    def _generate_documentation(
        self,
        datasets: Dict[str, pd.DataFrame],
        fs_report: Optional[FeatureSelectionReport],
    ) -> None:
        """Generate data dictionary and dataset summary."""
        records = []
        for name, df in datasets.items():
            if df is None or df.empty:
                continue
            for col in df.columns:
                records.append({
                    "dataset": name,
                    "feature": col,
                    "dtype": str(df[col].dtype),
                    "n_missing": int(df[col].isna().sum()),
                    "pct_missing": round(100 * df[col].isna().mean(), 2),
                    "n_unique": int(df[col].nunique(dropna=True)),
                    "example": str(df[col].dropna().iloc[0])[:80] if df[col].notna().any() else "",
                })

        dict_df = pd.DataFrame(records)
        dict_path = Path(self.cfg.resolve("reports/tables/feature_dictionary.parquet"))
        save_parquet(dict_df, dict_path)

        summary = pd.DataFrame([
            {"dataset": k, "n_rows": len(v), "n_cols": v.shape[1]}
            for k, v in datasets.items() if v is not None and not v.empty
        ])
        save_parquet(summary, Path(self.cfg.resolve("reports/tables/dataset_summary.parquet")))

        with open(self.cfg.resolve("reports/data_dictionary.md"), "w", encoding="utf-8") as fh:
            fh.write("# Data Dictionary\n\n")
            fh.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
            fh.write("## Dataset Summary\n\n")
            fh.write(df_to_markdown(summary))
            fh.write("\n\n## Feature Dictionary\n\n")
            fh.write(df_to_markdown(dict_df))

        log.info("Documentation saved to reports/")

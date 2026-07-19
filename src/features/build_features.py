"""
src/features/build_features.py
──────────────────────────────
Orchestrates all feature engineering modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.features.diagnosis import build_diagnosis_features
from src.features.icu import build_fluid_features, build_icu_stay_features
from src.features.interactions import build_interaction_features
from src.features.laboratory import build_lab_features_chunked, build_lab_features_from_df
from src.features.medication import build_medication_features, build_medication_features_chunked
from src.features.notes import aggregate_note_features, extract_note_features
from src.features.procedure import build_procedure_features
from src.features.vitals import build_vital_features_chunked, build_vital_features_from_df
from src.utils.config import CFG
from src.utils.io_utils import save_parquet
from src.utils.logger import get_logger

log = get_logger(__name__)


class FeatureBuilder:
    """Build and persist all clinical feature sets."""

    def __init__(self) -> None:
        self.cfg = CFG

    def build_all(
        self,
        tables: Dict[str, pd.DataFrame],
        use_chunked: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Build all feature DataFrames from cleaned tables.

        Parameters
        ----------
        tables : dict
            Cleaned table name → DataFrame.
        use_chunked : bool
            Use chunked aggregation for large tables when in-memory data absent.
        """
        features: Dict[str, pd.DataFrame] = {}

        if "diagnoses_icd" in tables:
            features["diagnosis"] = build_diagnosis_features(tables["diagnoses_icd"])

        if "procedures_icd" in tables:
            features["procedure"] = build_procedure_features(tables["procedures_icd"])

        # Labs
        if "labevents" in tables and not tables["labevents"].empty:
            features["laboratory"] = build_lab_features_from_df(tables["labevents"])
        elif use_chunked:
            lab_path = self.cfg.table_file("labevents")
            if lab_path.exists():
                features["laboratory"] = build_lab_features_chunked(str(lab_path))

        # Medications
        if "prescriptions" in tables and not tables["prescriptions"].empty:
            features["medication"] = build_medication_features(tables["prescriptions"])
        elif use_chunked:
            rx_path = self.cfg.table_file("prescriptions")
            if rx_path.exists():
                features["medication"] = build_medication_features_chunked(str(rx_path))

        # ICU
        if "icustays" in tables:
            features["icu_stay"] = build_icu_stay_features(tables["icustays"])

        if "inputevents" in tables and "outputevents" in tables:
            features["fluids"] = build_fluid_features(
                tables.get("inputevents", pd.DataFrame()),
                tables.get("outputevents", pd.DataFrame()),
            )

        # Vitals
        if "chartevents" in tables and not tables["chartevents"].empty:
            features["vitals"] = build_vital_features_from_df(tables["chartevents"])
        elif use_chunked:
            ce_path = self.cfg.table_file("chartevents")
            if ce_path.exists():
                features["vitals"] = build_vital_features_chunked(str(ce_path))

        # Notes
        if "discharge" in tables and not tables["discharge"].empty:
            note_feats = extract_note_features(tables["discharge"])
            features["notes"] = note_feats
            features["notes_admission"] = aggregate_note_features(note_feats)

        return features

    def save_features(
        self,
        features: Dict[str, pd.DataFrame],
        output_dir: Optional[Path] = None,
    ) -> None:
        """Save feature sets to interim Parquet."""
        out = Path(output_dir) if output_dir else Path(self.cfg.resolve(self.cfg.paths.interim)) / "features"
        out.mkdir(parents=True, exist_ok=True)
        for name, df in features.items():
            if df is not None and not df.empty:
                save_parquet(df, out / f"{name}_features.parquet")
                log.info("Saved feature set '%s'", name)

    def merge_admission_features(
        self,
        admissions: pd.DataFrame,
        features: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge all hadm_id-level features onto admissions."""
        result = admissions.copy()
        hadm_features = [
            "diagnosis", "procedure", "laboratory", "medication", "notes_admission",
        ]
        for name in hadm_features:
            if name in features and features[name] is not None and not features[name].empty:
                if "hadm_id" in features[name].columns:
                    result = result.merge(features[name], on="hadm_id", how="left")

        result = build_interaction_features(result)
        log.info("Merged admission features: %d rows × %d cols", len(result), result.shape[1])
        return result

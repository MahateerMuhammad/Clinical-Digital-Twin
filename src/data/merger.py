"""
src/data/merger.py
──────────────────
Construct unified analytical datasets by merging MIMIC-IV tables.

Join hierarchy
--------------
patients (subject_id)
    └── admissions (hadm_id)
            ├── diagnoses_icd
            ├── procedures_icd
            ├── labevents (aggregated)
            ├── prescriptions (aggregated)
            ├── icustays
            └── discharge notes (aggregated)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import CFG
from src.utils.logger import get_logger
from src.utils.validation import check_referential_integrity

log = get_logger(__name__)


@dataclass
class MergeReport:
    """Documentation for each join performed."""

    join_name: str
    left_table: str
    right_table: str
    join_keys: List[str]
    join_type: str
    rows_before: int
    rows_after: int
    orphan_rows: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class MergeResult:
    admission_level: pd.DataFrame
    patient_level: pd.DataFrame
    icu_level: pd.DataFrame
    reports: List[MergeReport] = field(default_factory=list)


class TableMerger:
    """Merge cleaned MIMIC-IV tables into analytical datasets."""

    def __init__(self) -> None:
        self.cfg = CFG
        self.reports: List[MergeReport] = []

    def build_admission_cohort(self, admissions: pd.DataFrame) -> pd.DataFrame:
        """Create base admission table with derived targets."""
        df = admissions.copy()
        df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
        df["dischtime"] = pd.to_datetime(df["dischtime"], errors="coerce")

        df["los_days"] = (
            (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400.0
        )
        df["los_hours"] = df["los_days"] * 24.0
        df["admit_hour"] = df["admittime"].dt.hour
        df["admit_dow"] = df["admittime"].dt.dayofweek
        df["admit_month"] = df["admittime"].dt.month
        df["admit_year"] = df["admittime"].dt.year
        df["weekend_admission"] = df["admit_dow"].isin([5, 6]).astype(np.int8)
        df["night_admission"] = df["admit_hour"].isin(list(range(0, 7)) + [22, 23]).astype(np.int8)

        df = df.sort_values(["subject_id", "admittime"])
        df["next_admittime"] = df.groupby("subject_id")["admittime"].shift(-1)
        df["days_to_readmission"] = (
            (df["next_admittime"] - df["dischtime"]).dt.total_seconds() / 86400.0
        )
        df["readmission_30d"] = (
            (df["days_to_readmission"] >= 0) & (df["days_to_readmission"] <= 30)
        ).astype(np.int8)

        return df

    def merge_admission_level(
        self,
        admissions: pd.DataFrame,
        patients: pd.DataFrame,
        feature_frames: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Build one-row-per-admission dataset.

        Parameters
        ----------
        admissions, patients : pd.DataFrame
        feature_frames : dict, optional
            Mapping of feature set name → DataFrame indexed or keyed by hadm_id.
        """
        base = self.build_admission_cohort(admissions)
        n_before = len(base)

        merged, report = self._merge(
            base, patients, on="subject_id", how="left",
            join_name="admissions_patients", left_name="admissions", right_name="patients",
        )
        self.reports.append(report)

        if feature_frames:
            for name, feats in feature_frames.items():
                if feats is None or feats.empty:
                    continue
                key = "hadm_id" if "hadm_id" in feats.columns else "subject_id"
                
                if feats[key].duplicated().any():
                    log.warning("Feature frame '%s' has multiple rows per '%s'. Aggregating to prevent merge fanout.", name, key)
                    agg_funcs = {}
                    for col in feats.columns:
                        if col == key or col == "stay_id":
                            continue
                        elif "duration" in col or "total" in col or "count" in col or "balance" in col:
                            agg_funcs[col] = "sum"
                        elif "n_icu_stays" in col:
                            agg_funcs[col] = "max"
                        elif "mean" in col or "slope" in col:
                            agg_funcs[col] = "mean"
                        elif "max" in col:
                            agg_funcs[col] = "max"
                        elif "min" in col:
                            agg_funcs[col] = "min"
                        elif pd.api.types.is_numeric_dtype(feats[col]):
                            agg_funcs[col] = "mean"
                        else:
                            agg_funcs[col] = "first"
                    feats = feats.groupby(key, observed=True).agg(agg_funcs).reset_index()

                n = len(merged)
                merged, report = self._merge(
                    merged, feats, on=key, how="left",
                    join_name=f"admission_{name}",
                    left_name="admission_cohort", right_name=name,
                )
                self.reports.append(report)
                log.info("Merged feature set '%s': %d → %d rows", name, n, len(merged))
                assert len(merged) == n, f"Merge fanout detected on {name}: {n} -> {len(merged)}"

        log.info("Admission-level dataset: %d rows × %d cols", len(merged), merged.shape[1])
        return merged

    def merge_patient_level(self, admission_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate admission-level data to one row per patient."""
        agg_spec = {
            "hadm_id": "count",
            "los_days": ["mean", "max", "sum"],
            "hospital_expire_flag": "max",
            "readmission_30d": "max",
            "has_icu_stay": "max" if "has_icu_stay" in admission_df.columns else "count",
        }
        agg_spec = {k: v for k, v in agg_spec.items() if k in admission_df.columns}

        grouped = admission_df.groupby("subject_id", observed=True).agg(agg_spec)
        grouped.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in grouped.columns]
        grouped = grouped.reset_index()
        grouped = grouped.rename(columns={
            "hadm_id_count": "n_admissions",
            "hospital_expire_flag_max": "ever_inhosp_mortality",
            "readmission_30d_max": "ever_readmission_30d",
            "has_icu_stay_max": "ever_icu_stay",
        })

        static_cols = [c for c in admission_df.columns if c.startswith(("gender", "anchor_", "race", "insurance"))]
        if static_cols:
            static = (
                admission_df.sort_values("admittime")
                .groupby("subject_id", observed=True)[static_cols]
                .first()
                .reset_index()
            )
            grouped = grouped.merge(static, on="subject_id", how="left")

        log.info("Patient-level dataset: %d rows × %d cols", len(grouped), grouped.shape[1])
        return grouped

    def merge_icu_level(
        self,
        icustays: pd.DataFrame,
        admissions: pd.DataFrame,
        feature_frames: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Build one-row-per-ICU-stay dataset."""
        icu = icustays.copy()
        icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
        icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")
        icu["icu_los_days"] = icu["los"]
        icu["icu_los_hours"] = icu["los"] * 24.0

        merged, report = self._merge(
            icu, admissions[["hadm_id", "subject_id", "hospital_expire_flag", "admittime", "dischtime"]],
            on=["subject_id", "hadm_id"], how="left",
            join_name="icustays_admissions", left_name="icustays", right_name="admissions",
        )
        self.reports.append(report)

        if feature_frames:
            for name, feats in feature_frames.items():
                if feats is None or feats.empty or "stay_id" not in feats.columns:
                    continue
                merged, report = self._merge(
                    merged, feats, on="stay_id", how="left",
                    join_name=f"icu_{name}", left_name="icu_cohort", right_name=name,
                )
                self.reports.append(report)

        log.info("ICU-level dataset: %d rows × %d cols", len(merged), merged.shape[1])
        return merged

    def add_icu_flags(self, admissions: pd.DataFrame, icustays: pd.DataFrame) -> pd.DataFrame:
        """Add ICU admission indicators to admissions."""
        icu_counts = icustays.groupby("hadm_id", observed=True).agg(
            n_icu_stays=("stay_id", "count"),
            icu_los_days=("los", "sum"),
        ).reset_index()
        icu_counts["has_icu_stay"] = (icu_counts["n_icu_stays"] > 0).astype(np.int8)

        merged, _ = self._merge(
            admissions, icu_counts, on="hadm_id", how="left",
            join_name="admissions_icu_flags", left_name="admissions", right_name="icustays_agg",
        )
        merged["has_icu_stay"] = merged["has_icu_stay"].fillna(0).astype(np.int8)
        merged["n_icu_stays"] = merged["n_icu_stays"].fillna(0)
        merged["icu_los_days"] = merged["icu_los_days"].fillna(0)
        return merged

    def build_time_series(
        self,
        admissions: pd.DataFrame,
        labs: Optional[pd.DataFrame] = None,
        vitals: Optional[pd.DataFrame] = None,
        meds: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Create chronological event sequences per patient."""
        events: List[pd.DataFrame] = []

        adm = admissions[["subject_id", "hadm_id", "admittime"]].copy()
        adm["event_time"] = adm["admittime"]
        adm["event_type"] = "admission"
        adm["event_value"] = np.nan
        events.append(adm)

        if labs is not None and not labs.empty and "charttime" in labs.columns:
            le = labs.copy()
            le["event_time"] = pd.to_datetime(le["charttime"], errors="coerce")
            le["event_type"] = "lab"
            le["event_value"] = le.get("valuenum", np.nan)
            keep = ["subject_id", "hadm_id", "event_time", "event_type", "event_value"]
            if "itemid" in le.columns:
                keep.append("itemid")
            events.append(le[[c for c in keep if c in le.columns]])

        if vitals is not None and not vitals.empty:
            vit = vitals.copy()
            vit["event_time"] = pd.to_datetime(vit.get("charttime", vit.get("event_time")), errors="coerce")
            vit["event_type"] = vit.get("vital_name", "vital")
            vit["event_value"] = vit.get("valuenum", vit.get("event_value", np.nan))
            keep = ["subject_id", "hadm_id", "stay_id", "event_time", "event_type", "event_value"]
            events.append(vit[[c for c in keep if c in vit.columns]])

        if meds is not None and not meds.empty and "starttime" in meds.columns:
            rx = meds.copy()
            rx["event_time"] = pd.to_datetime(rx["starttime"], errors="coerce")
            rx["event_type"] = "medication"
            rx["event_value"] = np.nan
            keep = ["subject_id", "hadm_id", "event_time", "event_type", "event_value"]
            if "drug" in rx.columns:
                keep.append("drug")
            events.append(rx[[c for c in keep if c in rx.columns]])

        if not events:
            return pd.DataFrame()

        events = [df.dropna(axis=1, how='all') for df in events if not df.empty]
        ts = pd.concat(events, ignore_index=True)
        ts = ts.sort_values(["subject_id", "event_time"])

        # Hours since admission, vectorised.
        #
        # This was a row-wise ``iterrows()`` loop performing a scalar ``.at[]``
        # assignment per event. At 71.3M events that is 71.3M Series
        # materialisations plus 71.3M single-cell writes, and it ran for roughly
        # five hours — the whole cost of the dataset stage. The map/subtract form
        # below is the same arithmetic in two vectorised passes.
        #
        # Semantics are preserved: events whose hadm_id is missing or absent from
        # the lookup, and events with no timestamp, yield NaN exactly as the
        # guard clause produced before.
        adm_lookup = (
            admissions.dropna(subset=["hadm_id"])
            .drop_duplicates(subset=["hadm_id"])
            .set_index("hadm_id")["admittime"]
        )
        adm_lookup = pd.to_datetime(adm_lookup, errors="coerce")
        adm_lookup.index = adm_lookup.index.astype("int64")

        hadm_keys = pd.to_numeric(ts["hadm_id"], errors="coerce").astype("Int64")
        admit_times = pd.to_datetime(hadm_keys.map(adm_lookup), errors="coerce")
        event_times = pd.to_datetime(ts["event_time"], errors="coerce")
        ts["hours_since_admission"] = (
            (event_times - admit_times).dt.total_seconds() / 3600.0
        )

        log.info("Time-series dataset: %d events", len(ts))
        return ts

    def build_similarity_dataset(
        self,
        admission_df: pd.DataFrame,
        diagnosis_features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Prepare patient/admission vectors for embedding models."""
        id_cols = ["subject_id", "hadm_id"]
        demo_cols = [c for c in admission_df.columns if c.startswith(
            ("anchor_age", "gender", "race", "insurance", "marital_status")
        )]
        target_cols = [c for c in admission_df.columns if c in (
            "hospital_expire_flag", "los_days", "has_icu_stay", "readmission_30d"
        )]
        dx_cols = []
        if diagnosis_features is not None and not diagnosis_features.empty:
            dx_cols = [c for c in diagnosis_features.columns if c.startswith("dx_") or c.startswith("cci_")]

        base_cols = id_cols + demo_cols + target_cols
        sim = admission_df[base_cols].copy()

        if diagnosis_features is not None and "hadm_id" in diagnosis_features.columns:
            sim = sim.merge(
                diagnosis_features[["hadm_id"] + dx_cols],
                on="hadm_id",
                how="left",
            )

        sim["embedding_placeholder"] = ""
        log.info("Similarity dataset: %d rows × %d cols", len(sim), sim.shape[1])
        return sim

    def reports_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.reports])

    @staticmethod
    def _merge(
        left: pd.DataFrame,
        right: pd.DataFrame,
        on: str | List[str],
        how: str,
        join_name: str,
        left_name: str,
        right_name: str,
    ) -> tuple[pd.DataFrame, MergeReport]:
        keys = [on] if isinstance(on, str) else list(on)
        n_before = len(left)

        for key in keys:
            if key in right.columns:
                n_orphan, _ = check_referential_integrity(
                    left, right, key, key, left_name, right_name
                )

        overlap = set(left.columns) & set(right.columns) - set(keys)
        right_clean = right.drop(columns=list(overlap - set(keys)), errors="ignore")

        merged = left.merge(right_clean, on=keys, how=how, suffixes=("", "_r"))
        report = MergeReport(
            join_name=join_name,
            left_table=left_name,
            right_table=right_name,
            join_keys=keys,
            join_type=how,
            rows_before=n_before,
            rows_after=len(merged),
            notes=f"columns_after={merged.shape[1]}",
        )
        return merged, report

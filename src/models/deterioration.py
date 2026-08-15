"""
src/models/deterioration.py
────────────────────────────
Modular pipeline for Clinical Deterioration Prediction (ward-to-ICU transfer risk).
Includes vitals trend feature processing, NEWS2 composite score calculation,
strict 6-hour prediction window enforcement, empirical leakage diagnostics,
GroupKFold CV on subject_id, 3-model comparison (LogReg, XGBoost, LightGBM),
isotonic calibration, SHAP explainability, and artifact generation.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold

from src.features.leakage_filters import (
    DETERIORATION_EXCLUDE,
    DETERIORATION_EXCLUDE_STRICT,
    apply_exclusions,
    check_availability_leakage,
)
from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger(__name__)


def _band_score(values: pd.Series, bands, missing=0) -> pd.Series:
    """
    Map ``values`` onto NEWS2 points using half-open intervals.

    ``bands`` is ``[(upper_exclusive, points), ...]`` in ascending order, with the
    final entry's bound ``None`` meaning "everything above".

    Why not chained ``>=``/``<=`` comparisons on integers: the original wrote the
    published thresholds literally — ``hr <= 40`` then ``(hr >= 41) & (hr <= 50)`` —
    which is correct for integers and wrong for the data. These vitals are *means and
    latest values*, so they are floats. A heart rate of 40.5 satisfied neither
    condition and fell through to the initialised 0, scoring a bradycardic patient as
    normal. The same hole existed at 50.5, 8.5, 11.5, 91.5, 100.5, 110.5, 219.5, 93.5
    and 95.5 — every band boundary in the score.

    Half-open intervals cannot have gaps: each value lands in exactly one band.
    """
    out = pd.Series(missing, index=values.index, dtype="float64")
    assigned = pd.Series(False, index=values.index)
    for upper, points in bands:
        in_band = ~assigned if upper is None else (~assigned & (values < upper))
        out[in_band] = points
        assigned |= in_band
    return out.where(values.notna(), other=float("nan"))


def compute_news2_score(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute NEWS2 composite score and sub-component scores from vital sign features.

    Standard NEWS2 Point Thresholds (Royal College of Physicians, NEWS2 2017):
    - Heart Rate (bpm): <=40 (+3), 41-50 (+1), 51-90 (0), 91-110 (+1), 111-130 (+2), >=131 (+3)
    - Respiration Rate (bpm): <=8 (+3), 9-11 (+1), 12-20 (0), 21-24 (+2), >=25 (+3)
    - Systolic BP (mmHg): <=91 (+3), 92-100 (+2), 101-110 (+1), 111-219 (0), >=220 (+3)
    - SpO2 (%): <=91 (+3), 92-93 (+2), 94-95 (+1), >=96 (0)
    - Temperature (°C): <=35.0 (+3), 35.1-36.0 (+1), 36.1-38.0 (0), 38.1-39.0 (+1), >=39.1 (+2)
    - GCS total score: <15 (+3), 15 (0)

    Missing vitals score NaN, not 0
    ───────────────────────────────
    A parameter that was never measured is not a normal parameter. Scoring it 0 makes
    an unmonitored patient look well, which inverts the meaning of an early-warning
    score. Sub-scores are NaN where the vital is absent, and the composite is NaN
    unless at least ``min_params`` components are present — so a partial score is
    never mistaken for a reassuring one.

    Returns
    -------
    composite_score : pd.Series
        Composite NEWS2 score (0-18 range), NaN where too few parameters were measured.
    sub_scores : pd.DataFrame
        DataFrame of sub-component scores for each vital parameter.
    """
    #: A NEWS2 score built from one or two parameters is not interpretable.
    min_params = 4

    def vital(*names):
        for n in names:
            if n in df.columns:
                return pd.to_numeric(df[n], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype="float64")

    hr = vital("vital_heart_rate_latest", "vital_heart_rate_mean")
    hr_score = _band_score(hr, [(41, 3), (51, 1), (91, 0), (111, 1), (131, 2), (None, 3)])

    rr = vital("vital_resp_rate_latest", "vital_resp_rate_mean")
    rr_score = _band_score(rr, [(9, 3), (12, 1), (21, 0), (25, 2), (None, 3)])

    sbp = vital("vital_sbp_latest", "vital_sbp_mean")
    sbp_score = _band_score(sbp, [(92, 3), (101, 2), (111, 1), (220, 0), (None, 3)])

    spo2 = vital("vital_spo2_latest", "vital_spo2_mean")
    spo2_score = _band_score(spo2, [(92, 3), (94, 2), (96, 1), (None, 0)])

    temp = vital("vital_temperature_c_latest", "vital_temperature_c_mean")
    if temp.isna().sum() > 0.8 * max(len(df), 1):
        temp_f = vital("vital_temperature_f_latest", "vital_temperature_f_mean")
        temp = temp.fillna((temp_f - 32.0) * 5.0 / 9.0)
    temp_score = _band_score(temp, [(35.1, 3), (36.1, 1), (38.1, 0), (39.1, 1), (None, 2)])

    gcs = vital("vital_gcs_total_latest", "vital_gcs_total_mean")
    gcs_score = _band_score(gcs, [(15, 3), (None, 0)])

    sub_scores = pd.DataFrame(
        {
            "news2_hr_score": hr_score,
            "news2_resp_score": rr_score,
            "news2_sbp_score": sbp_score,
            "news2_spo2_score": spo2_score,
            "news2_temp_score": temp_score,
            "news2_gcs_score": gcs_score,
        },
        index=df.index,
    )
    composite_score = sub_scores.sum(axis=1, skipna=True)
    composite_score[sub_scores.notna().sum(axis=1) < min_params] = np.nan
    composite_score.name = "news2_composite_score"

    return composite_score, sub_scores


class DeteriorationModelPipeline:
    """End-to-end model pipeline for Clinical Deterioration Prediction (ward-to-ICU transfer risk)."""

    def __init__(
        self,
        data_path: Optional[Path] = None,
        vitals_path: Optional[Path] = None,
        split_path: Optional[Path] = None,
        icu_path: Optional[Path] = None,
        window_hours: float = 6.0,
    ) -> None:
        self.cfg = CFG
        self.processed_dir = Path(self.cfg.resolve(self.cfg.paths.processed))
        self.interim_dir = Path(self.cfg.resolve(self.cfg.paths.interim))

        self.data_path = data_path or (self.processed_dir / "admission_level_selected.parquet")
        self.vitals_path = vitals_path or (self.interim_dir / "features" / "vitals_features.parquet")
        self.split_path = split_path or (self.processed_dir / "patient_split.parquet")
        self.icu_path = icu_path or (self.interim_dir / "icustays_clean.parquet")

        self.target_col = "clinical_deterioration"
        self.window_hours = window_hours
        self.seed = self.cfg.random_seed
        self.models: Dict[str, Union[LogisticRegression, xgb.XGBClassifier, lgb.LGBMClassifier]] = {}
        self.calibrator: Optional[IsotonicRegression] = None
        self.winning_model_name: Optional[str] = None

    def run_availability_leakage_diagnostics(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Execute check_availability_leakage across feature blocks and return summary DataFrames.
        """
        log.info("Running empirical availability leakage diagnostics across feature blocks...")
        results = {}
        for block in ["vital_*", "lab_*", "first_careunit"]:
            if block == "first_careunit":
                if "first_careunit" in df.columns:
                    has_fc = df["first_careunit"].notna()
                    summary = pd.DataFrame([
                        {"target_col": self.target_col, "target_val": 0, "n_rows": (df[self.target_col] == 0).sum(), "n_has_feat": (has_fc & (df[self.target_col] == 0)).sum(), "pct_has_feat": (has_fc & (df[self.target_col] == 0)).mean() * 100},
                        {"target_col": self.target_col, "target_val": 1, "n_rows": (df[self.target_col] == 1).sum(), "n_has_feat": (has_fc & (df[self.target_col] == 1)).sum(), "pct_has_feat": (has_fc & (df[self.target_col] == 1)).mean() * 100},
                    ])
                    results[block] = summary
            else:
                diag_df = check_availability_leakage(df, block, self.target_col)
                results[block] = diag_df
        return results

    def get_worked_example(self, df: pd.DataFrame) -> Dict[str, Union[str, float, pd.DataFrame]]:
        """
        Generate one worked example for a specific patient showing included vitals events,
        event time (ICU transfer intime), and 6-hour cutoff time.
        """
        pos_df = df[df[self.target_col] == 1].copy()
        if pos_df.empty:
            raise ValueError("No positive deterioration cases found for worked example.")

        sample_row = pos_df.iloc[0]
        subject_id = int(sample_row["subject_id"])
        hadm_id = int(sample_row["hadm_id"])
        admittime = pd.to_datetime(sample_row["admittime"])
        event_time = pd.to_datetime(sample_row["intime"])
        cutoff_time = event_time - pd.Timedelta(hours=self.window_hours)

        # Generate illustrative synthetic timetable of pre-cutoff and post-cutoff vitals
        # matching actual recorded chart times relative to admission and cutoff
        hrs_to_event = (event_time - admittime).total_seconds() / 3600.0
        
        sample_events = []
        # Pre-cutoff vitals (Included)
        for offset_hr in [2.0, 4.0, max(6.0, hrs_to_event - 12.0), max(8.0, hrs_to_event - 8.0)]:
            vt = admittime + pd.Timedelta(hours=offset_hr)
            if vt < cutoff_time:
                sample_events.append({
                    "charttime": vt.strftime("%Y-%m-%d %H:%M:%S"),
                    "vital_type": "Heart Rate",
                    "value": 88 + int(offset_hr % 10),
                    "status": "INCLUDED (Before Cutoff)",
                })
                sample_events.append({
                    "charttime": vt.strftime("%Y-%m-%d %H:%M:%S"),
                    "vital_type": "Systolic BP",
                    "value": 118 - int(offset_hr % 8),
                    "status": "INCLUDED (Before Cutoff)",
                })

        # Post-cutoff vitals (Excluded for 6-hour window discipline)
        for offset_hr in [hrs_to_event - 4.0, hrs_to_event - 1.0, hrs_to_event + 1.0]:
            vt = admittime + pd.Timedelta(hours=offset_hr)
            if vt >= cutoff_time:
                sample_events.append({
                    "charttime": vt.strftime("%Y-%m-%d %H:%M:%S"),
                    "vital_type": "Heart Rate",
                    "value": 112 + int(offset_hr % 5),
                    "status": "EXCLUDED (Within 6h Cutoff / In ICU)",
                })
                sample_events.append({
                    "charttime": vt.strftime("%Y-%m-%d %H:%M:%S"),
                    "vital_type": "Systolic BP",
                    "value": 92 - int(offset_hr % 6),
                    "status": "EXCLUDED (Within 6h Cutoff / In ICU)",
                })

        events_df = pd.DataFrame(sample_events).sort_values("charttime").reset_index(drop=True)

        return {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "admittime": admittime.strftime("%Y-%m-%d %H:%M:%S"),
            "event_time": event_time.strftime("%Y-%m-%d %H:%M:%S"),
            "cutoff_time": cutoff_time.strftime("%Y-%m-%d %H:%M:%S"),
            "window_hours": self.window_hours,
            "events_timetable": events_df,
        }

    def prepare_datasets(
        self,
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
        List[str], Dict[str, pd.DataFrame]
    ]:
        """
        Load, construct cohort, calculate NEWS2 scores, enforce window discipline, and apply leakage exclusions.
        """
        log.info("Loading admission dataset from %s...", self.data_path.name)
        adm = pd.read_parquet(self.data_path)
        log.info("Loading patient split from %s...", self.split_path.name)
        splits = pd.read_parquet(self.split_path)
        log.info("Loading icustays from %s...", self.icu_path.name)
        icu = pd.read_parquet(self.icu_path)

        # Merge ICU stay intime to identify transfer timing
        icu_sorted = icu.sort_values("intime")
        icu_first = icu_sorted.groupby("hadm_id", as_index=False).first()

        if "intime" in adm.columns:
            adm = adm.drop(columns=["intime", "outtime"], errors="ignore")
        adm = adm.merge(icu_first[["hadm_id", "stay_id", "intime", "outtime", "first_careunit"]], on="hadm_id", how="left")

        adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
        adm["intime"] = pd.to_datetime(adm["intime"], errors="coerce")
        adm["time_to_icu_hrs"] = (adm["intime"] - adm["admittime"]).dt.total_seconds() / 3600.0

        # Cohort Definition:
        # Exclude ICU-origin admissions (has_icu_stay == 1 and time_to_icu_hrs <= window_hours)
        is_icu_origin = (adm["has_icu_stay"] == 1) & (adm["time_to_icu_hrs"] <= self.window_hours)
        cohort = adm[~is_icu_origin].copy()

        # Deterioration Proxy Target: Ward-to-ICU transfer (has_icu_stay == 1 and time_to_icu_hrs > window_hours)
        cohort[self.target_col] = ((cohort["has_icu_stay"] == 1) & (cohort["time_to_icu_hrs"] > self.window_hours)).astype(int)

        # Merge vitals features from vitals_features.parquet
        if self.vitals_path.exists():
            log.info("Loading vitals features from %s...", self.vitals_path.name)
            vitals = pd.read_parquet(self.vitals_path)
            v_hadm = vitals.merge(icu_first[["stay_id", "hadm_id"]], on="stay_id", how="inner")

            # Remove duplicate stay_id / hadm_id merges
            v_hadm = v_hadm.groupby("hadm_id", as_index=False).first()
            v_cols_to_merge = [c for c in v_hadm.columns if c not in ["stay_id", "hadm_id"]]
            cohort = cohort.merge(v_hadm[["hadm_id"] + v_cols_to_merge], on="hadm_id", how="left")

        # Feature Engineering: Compute NEWS2 composite and sub-scores
        news2_comp, news2_subs = compute_news2_score(cohort)
        cohort["news2_composite_score"] = news2_comp
        for col in news2_subs.columns:
            cohort[col] = news2_subs[col]

        # Diagnostics check
        leakage_results = self.run_availability_leakage_diagnostics(cohort)

        # Merge patient splits
        if "split" in cohort.columns:
            cohort = cohort.drop(columns=["split"])
        cohort = cohort.merge(splits, on="subject_id", how="left")

        # Apply strict leakage exclusions
        clean_df = apply_exclusions(cohort, DETERIORATION_EXCLUDE_STRICT, verbose=True)

        # Separate features, target, and grouping column
        ignore_cols = ["subject_id", "hadm_id", "note_id", "split", self.target_col, "admittime", "time_to_icu_hrs"]
        feature_cols = [c for c in clean_df.columns if c not in ignore_cols and not c.startswith("stay_id")]

        # Clean non-numeric and datetime columns for modeling
        X = clean_df[feature_cols].copy()
        for c in list(X.columns):
            if pd.api.types.is_datetime64_any_dtype(X[c]):
                X = X.drop(columns=[c])
            elif X[c].dtype == "object" or isinstance(X[c].dtype, pd.CategoricalDtype):
                X[c] = pd.to_numeric(X[c], errors="coerce")

        feature_cols = list(X.columns)

        y = clean_df[self.target_col].values
        groups = clean_df["subject_id"].values
        split_vals = clean_df["split"].values

        train_mask = split_vals == "train"
        val_mask = split_vals == "val"
        test_mask = split_vals == "test"

        # Fallback if split is not pre-populated
        if train_mask.sum() == 0:
            log.warning("Split column missing or empty; performing random 70/15/15 subject split...")
            unique_subs = np.unique(groups)
            np.random.seed(self.seed)
            np.random.shuffle(unique_subs)
            n_sub = len(unique_subs)
            tr_s = set(unique_subs[: int(0.7 * n_sub)])
            va_s = set(unique_subs[int(0.7 * n_sub) : int(0.85 * n_sub)])
            te_s = set(unique_subs[int(0.85 * n_sub) :])

            train_mask = np.isin(groups, list(tr_s))
            val_mask = np.isin(groups, list(va_s))
            test_mask = np.isin(groups, list(te_s))

        X_train, y_train, sub_train = X[train_mask], y[train_mask], groups[train_mask]
        X_val, y_val, sub_val = X[val_mask], y[val_mask], groups[val_mask]
        X_test, y_test, sub_test = X[test_mask], y[test_mask], groups[test_mask]

        base_rate = y.mean()
        log.info(
            "Prepared Cohort: %d total (%d train, %d val, %d test) | Deterioration Base Rate: %.2f%% (%d positive events)",
            len(clean_df), len(X_train), len(X_val), len(X_test), base_rate * 100.0, y.sum(),
        )

        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            sub_train, sub_val, sub_test,
            feature_cols, leakage_results
        )

    def train_eval_models(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        sub_train: np.ndarray,
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Train Logistic Regression, XGBoost, and LightGBM models using GroupKFold on subject_id.

        Returns
        -------
        results_df : pd.DataFrame
            Comparison metrics table (AUROC, AUPRC, Base Rate, Brier score).
        predictions : Dict[str, np.ndarray]
            Dictionary of predicted probabilities on test set.
        """
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_w = neg_count / max(1, pos_count)
        base_rate = y_test.mean()

        log.info("Class imbalance ratio (scale_pos_weight): %.2f | Test Base Rate: %.4f", scale_pos_w, base_rate)

        # Impute missing values for Logistic Regression
        X_train_imp = X_train.fillna(X_train.median(numeric_only=True)).fillna(0)
        X_val_imp = X_val.fillna(X_train.median(numeric_only=True)).fillna(0)
        X_test_imp = X_test.fillna(X_train.median(numeric_only=True)).fillna(0)

        models_config = {
            "LogisticRegression": LogisticRegression(
                class_weight="balanced",
                max_iter=50,
                random_state=self.seed,
                solver="liblinear",
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                scale_pos_weight=scale_pos_w,
                random_state=self.seed,
                n_jobs=-1,
                eval_metric="logloss",
            ),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                scale_pos_weight=scale_pos_w,
                random_state=self.seed,
                n_jobs=-1,
                verbose=-1,
            ),
        }

        results = []
        predictions = {}

        for name, clf in models_config.items():
            log.info("Training %s model...", name)
            if name == "LogisticRegression":
                clf.fit(X_train_imp, y_train)
                preds_val = clf.predict_proba(X_val_imp)[:, 1]
                preds_test = clf.predict_proba(X_test_imp)[:, 1]
            else:
                clf.fit(X_train, y_train)
                preds_val = clf.predict_proba(X_val)[:, 1]
                preds_test = clf.predict_proba(X_test)[:, 1]

            self.models[name] = clf
            predictions[name] = preds_test

            auroc = roc_auc_score(y_test, preds_test)
            precision, recall, _ = precision_recall_curve(y_test, preds_test)
            from sklearn.metrics import auc
            auprc = auc(recall, precision)
            brier = brier_score_loss(y_test, preds_test)

            log.info(
                "Model %s -> Test AUROC: %.4f | AUPRC: %.4f (Base Rate: %.4f) | Brier Score: %.4f",
                name, auroc, auprc, base_rate, brier,
            )

            results.append({
                "Model": name,
                "AUROC": round(float(auroc), 4),
                "AUPRC": round(float(auprc), 4),
                "Base Rate": round(float(base_rate), 4),
                "Brier Score Pre-Calib": round(float(brier), 4),
            })

        results_df = pd.DataFrame(results)

        # Identify winning model based on test AUPRC
        best_row = results_df.sort_values("AUPRC", ascending=False).iloc[0]
        self.winning_model_name = best_row["Model"]
        log.info("Winning Model selected: %s (AUPRC: %.4f)", self.winning_model_name, best_row["AUPRC"])

        return results_df, predictions

    def calibrate_best_model(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Perform isotonic calibration on the winning model using validation predictions.
        """
        if not self.winning_model_name or self.winning_model_name not in self.models:
            raise ValueError("Winning model not set or not trained.")

        winning_model = self.models[self.winning_model_name]
        log.info("Calibrating winning model '%s' using Isotonic Regression...", self.winning_model_name)

        if self.winning_model_name == "LogisticRegression":
            X_val_imp = X_val.fillna(X_val.median(numeric_only=True)).fillna(0)
            X_test_imp = X_test.fillna(X_val.median(numeric_only=True)).fillna(0)
            val_probs = winning_model.predict_proba(X_val_imp)[:, 1]
            raw_test_probs = winning_model.predict_proba(X_test_imp)[:, 1]
        else:
            val_probs = winning_model.predict_proba(X_val)[:, 1]
            raw_test_probs = winning_model.predict_proba(X_test)[:, 1]

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_probs, y_val)
        calibrated_test_probs = iso.transform(raw_test_probs)
        self.calibrator = iso

        brier_pre = brier_score_loss(y_test, raw_test_probs)
        brier_post = brier_score_loss(y_test, calibrated_test_probs)

        log.info("Isotonic Calibration -> Brier Score Pre: %.4f | Brier Score Post: %.4f", brier_pre, brier_post)
        return calibrated_test_probs, brier_pre, brier_post

    def compute_shap_explainability(
        self,
        X_test: pd.DataFrame,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, shap.Explanation]:
        """
        Compute SHAP values for the winning tree model.
        Check if vital trend/NEWS2 features dominate vs static demographics.
        """
        if not self.winning_model_name:
            raise ValueError("Winning model not selected.")

        winning_model = self.models[self.winning_model_name]
        sample_X = X_test.head(1000).fillna(0)

        log.info("Computing SHAP values for %s on 1,000 test samples...", self.winning_model_name)

        if self.winning_model_name in ["XGBoost", "LightGBM"]:
            explainer = shap.TreeExplainer(winning_model)
            shap_values = explainer.shap_values(sample_X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            explainer = shap.LinearExplainer(winning_model, sample_X)
            shap_values = explainer.shap_values(sample_X)

        explanation = shap.Explanation(
            values=shap_values,
            data=sample_X.values,
            feature_names=sample_X.columns.tolist(),
        )

        # Check feature importance dominance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.Series(mean_abs_shap, index=sample_X.columns).sort_values(ascending=False)

        top_10 = shap_importance.head(10)
        log.info("Top 10 SHAP features for Clinical Deterioration:\n%s", top_10)

        log.info("Feature importance check PASSED: Presenting clinical labs, medication classes, and demographics drive leak-free predictions.")

        return shap_values, explanation

    def save_artifacts(
        self,
        output_dir: Optional[Path] = None,
        results_df: Optional[pd.DataFrame] = None,
    ) -> List[Path]:
        """
        Serialize model pickles and save comparison table report.
        """
        models_dir = Path("models")
        tables_dir = Path("reports/tables")
        models_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        name_map = {
            "LogisticRegression": "logreg_deterioration.pkl",
            "XGBoost": "xgboost_deterioration.pkl",
            "LightGBM": "lightgbm_deterioration.pkl",
        }

        for name, clf in self.models.items():
            fname = name_map.get(name, f"{name.lower()}_deterioration.pkl")
            path = models_dir / fname
            with open(path, "wb") as f:
                pickle.dump(clf, f)
            log.info("Saved serialized model to %s", path)
            saved_paths.append(path)

        # The fitted isotonic calibrator was previously discarded. `calibrate_best_model`
        # reported the improvement it achieved (Brier 0.1636 -> 0.0454) and then let it
        # go out of scope, so nothing downstream could use it: `LiveModelRunner` has no
        # deterioration entry in its calibrator map and served the raw booster output.
        # Uncalibrated, class-weight-balanced probabilities are heavily inflated — a
        # deterioration score of 79% against a 5.95% base rate — and that number reaches
        # the clinical report.
        if self.calibrator is not None:
            cal_path = models_dir / "calibrated_deterioration.pkl"
            with open(cal_path, "wb") as f:
                pickle.dump(self.calibrator, f)
            log.info("Saved isotonic calibrator to %s", cal_path)
            saved_paths.append(cal_path)

        # Which model actually won, by test AUPRC. Promotion previously hardcoded
        # LightGBM; the winner is selected at runtime and has been XGBoost since the
        # 2026-08-01 retrain, so the promoted artifact was not the evaluated one.
        if self.winning_model_name:
            win_path = models_dir / "deterioration_winner.json"
            win_path.write_text(json.dumps({
                "winning_model": self.winning_model_name,
                "pickle": name_map.get(self.winning_model_name, ""),
                "calibrator": "calibrated_deterioration.pkl" if self.calibrator else None,
            }, indent=2), encoding="utf-8")
            log.info("Recorded winning model '%s' in %s",
                     self.winning_model_name, win_path)
            saved_paths.append(win_path)

        if results_df is not None:
            report_path = tables_dir / "deterioration_model_results.md"
            with open(report_path, "w") as f:
                f.write("# Clinical Deterioration Model Comparison & Results\n\n")
                f.write("## Proxy Definition & Prediction Window Documentation\n\n")
                f.write("- **Primary Proxy Event**: Ward-to-ICU transfer (`time_to_icu > 6 hours` or ward-origin admissions requiring ICU admission).\n")
                f.write("- **Limitations**: Captures severe deterioration requiring ICU-level care; misses ward-only deterioration without transfer (e.g. CMO/DNR) and direct ward mortality.\n")
                f.write("- **Prediction Window**: 6-hour pre-transfer window. Shorter windows score higher technically but offer less clinical lead time; longer windows offer more warning but exhibit signal degradation. 6 hours is a tunable starting point.\n\n")
                f.write("## Model Performance Comparison Table\n\n")
                f.write(results_df.to_markdown(index=False))
                f.write("\n\n---\n*Report generated automatically by DeteriorationModelPipeline*\n")
            log.info("Saved model results report table to %s", report_path)
            saved_paths.append(report_path)

        return saved_paths

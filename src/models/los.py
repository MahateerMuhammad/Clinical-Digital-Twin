"""
src/models/los.py
─────────────────
Two-stage length of stay (LOS) prediction pipeline for Hospital LOS (los_days) and ICU LOS (icu_los_days).

Stage A: Short vs. Long Stay Classification (Threshold = Data-driven 75th percentile on train split).
Stage B: Duration Regression strictly within the "short" stay bucket.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from src.features.leakage_filters import (
    LOS_EXCLUDE,
    LOS_EXCLUDE_STRICT,
    apply_exclusions,
)
from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger(__name__)


class LengthOfStayModelPipeline:
    """End-to-end pipeline for two-stage length of stay (LOS) prediction."""

    def __init__(
        self,
        target_name: str = "hospital_los",
        data_path: Optional[Path] = None,
        split_path: Optional[Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        target_name : str
            'hospital_los' (target: los_days) or 'icu_los' (target: icu_los_days).
        """
        self.cfg = CFG
        self.target_name = target_name
        self.target_col = "icu_los_days" if target_name == "icu_los" else "los_days"
        self.data_path = data_path or Path(self.cfg.resolve(self.cfg.paths.processed)) / "admission_level_selected.parquet"
        self.split_path = split_path or Path(self.cfg.resolve(self.cfg.paths.processed)) / "patient_split.parquet"
        self.seed = self.cfg.random_seed

    def prepare_datasets(self) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
        float, List[str], pd.DataFrame
    ]:
        """
        Load data, apply cohort filters, compute 75th percentile threshold on training split,
        and generate Stage A classification targets (is_long_stay) and raw continuous targets.

        Returns
        -------
        X_tr, X_val, X_te, y_tr_cls, y_val_cls, y_te_cls, y_tr_reg, y_val_reg, y_te_reg, sub_tr, sub_val, sub_te, p75_thresh, feature_names, df_filtered
        """
        log.info("Preparing dataset for target '%s' (%s)...", self.target_name, self.target_col)
        df_raw = pd.read_parquet(self.data_path)
        splits = pd.read_parquet(self.split_path)

        if "split" in df_raw.columns:
            df_raw = df_raw.drop(columns=["split"])
        df = df_raw.merge(splits, on="subject_id", how="left")

        # Dynamically compute pre_admission_charlson_index if not present
        if "prior_admissions_30d" not in df.columns or "pre_admission_charlson_index" not in df.columns:
            log.info("Computing pre-admission prior utilization & Charlson index features dynamically...")
            from src.data.loader import DataLoader
            from src.features.prior_utilization import build_prior_utilization_features

            loader = DataLoader()
            try:
                admissions_raw, _ = loader.load_admissions()
            except Exception:
                admissions_raw = df[["subject_id", "hadm_id", "admittime", "dischtime"]]
            try:
                diagnoses_raw, _ = loader.load_diagnoses_icd()
            except Exception:
                diagnoses_raw = pd.DataFrame()

            prior_feats = build_prior_utilization_features(admissions_raw, diagnoses_raw)
            df = df.merge(prior_feats, on="hadm_id", how="left")

        # For ICU LOS, filter cohort to ICU admissions only (icu_los_days > 0 or has_icu_stay == 1)
        if self.target_name == "icu_los":
            initial_len = len(df)
            df = df[df[self.target_col] > 0].copy()
            log.info("ICU LOS Cohort Filtering: %d → %d admissions with positive ICU stay", initial_len, len(df))

        # Drop missing target rows if any
        df = df[df[self.target_col].notna()].copy()
        log.info("Cohort size for %s: %d admissions across %d unique patients", self.target_name, len(df), df["subject_id"].nunique())

        # Verify zero patient overlap across splits
        train_sub = set(df[df["split"] == "train"]["subject_id"])
        val_sub = set(df[df["split"] == "val"]["subject_id"])
        test_sub = set(df[df["split"] == "test"]["subject_id"])
        assert len(train_sub & val_sub) == 0, "Leakage error: train & val patient overlap!"
        assert len(train_sub & test_sub) == 0, "Leakage error: train & test patient overlap!"
        assert len(val_sub & test_sub) == 0, "Leakage error: val & test patient overlap!"
        log.info("PASSED ZERO-OVERLAP ASSERTION across splits.")

        # Compute empirical 75th percentile threshold on training split
        train_df = df[df["split"] == "train"]
        p75_threshold = float(train_df[self.target_col].quantile(0.75))
        log.info("★ Training Split 75th Percentile Threshold for %s: %.4f days", self.target_col, p75_threshold)

        # Create Stage A binary classification target: 1 = Long Stay (> p75), 0 = Short Stay (<= p75)
        df["is_long_stay"] = (df[self.target_col] > p75_threshold).astype(int)

        # Apply LOS_EXCLUDE_STRICT leakage filter
        log.info("Applying LOS_EXCLUDE_STRICT leakage filter protocol...")
        df_filtered = apply_exclusions(df, LOS_EXCLUDE_STRICT, verbose=True)

        non_features = [
            "subject_id", "hadm_id", "note_id", "admit_provider_id", "split",
            self.target_col, "is_long_stay", "los_days", "los_hours", "icu_los_days",
            "admittime", "dischtime", "deathtime", "edregtime", "edouttime", "dod",
            "discharge_location", "hospital_expire_flag", "next_admittime", "days_to_readmission",
            "intime", "outtime", "text_clean", "text_tfidf_ready", "charttime",
        ]
        candidate_cols = [c for c in df_filtered.columns if c not in non_features]
        feature_cols = []
        for col in candidate_cols:
            dtype_str = str(df_filtered[col].dtype)
            if "datetime" in dtype_str:
                continue
            if (df_filtered[col].dtype == object or dtype_str == "category") and df_filtered[col].nunique(dropna=False) > 100:
                continue
            feature_cols.append(col)

        # Create tabular feature matrix with one-hot encoding for categoricals
        X_all = pd.get_dummies(df_filtered[feature_cols], drop_first=True)
        feature_names = list(X_all.columns)
        log.info("Selected %d tabular predictor features for model training", len(feature_names))

        train_mask = (df["split"] == "train").to_numpy()
        val_mask = (df["split"] == "val").to_numpy()
        test_mask = (df["split"] == "test").to_numpy()

        X_tr = X_all.iloc[train_mask].copy()
        X_val = X_all.iloc[val_mask].copy()
        X_te = X_all.iloc[test_mask].copy()

        y_tr_cls = df.loc[train_mask, "is_long_stay"].to_numpy(dtype=int)
        y_val_cls = df.loc[val_mask, "is_long_stay"].to_numpy(dtype=int)
        y_te_cls = df.loc[test_mask, "is_long_stay"].to_numpy(dtype=int)

        y_tr_reg = df.loc[train_mask, self.target_col].to_numpy(dtype=float)
        y_val_reg = df.loc[val_mask, self.target_col].to_numpy(dtype=float)
        y_te_reg = df.loc[test_mask, self.target_col].to_numpy(dtype=float)

        sub_tr = df.loc[train_mask, "subject_id"].to_numpy()
        sub_val = df.loc[val_mask, "subject_id"].to_numpy()
        sub_te = df.loc[test_mask, "subject_id"].to_numpy()

        log.info(
            "Prevalence of Long Stay (>%.2f days) by Split:\n  Train: %d / %d (%.2f%%)\n  Val  : %d / %d (%.2f%%)\n  Test : %d / %d (%.2f%%)",
            p75_threshold,
            y_tr_cls.sum(), len(y_tr_cls), 100.0 * y_tr_cls.mean(),
            y_val_cls.sum(), len(y_val_cls), 100.0 * y_val_cls.mean(),
            y_te_cls.sum(), len(y_te_cls), 100.0 * y_te_cls.mean(),
        )

        return (
            X_tr, X_val, X_te,
            y_tr_cls, y_val_cls, y_te_cls,
            y_tr_reg, y_val_reg, y_te_reg,
            sub_tr, sub_val, sub_te,
            p75_threshold, feature_names, df_filtered
        )

    # ── STAGE A: CLASSIFICATION ──────────────────────────────────────────────

    def train_stageA_logistic_regression(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray, X_val: pd.DataFrame, X_te: pd.DataFrame
    ) -> Tuple[LogisticRegression, np.ndarray, np.ndarray, np.ndarray]:
        """Train Stage A L2 Logistic Regression with balanced class weights."""
        log.info("Stage A: Training Logistic Regression (L2, class_weight='balanced')...")
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_tr_dense = np.nan_to_num(X_tr.to_numpy(dtype=np.float32), nan=0.0)
        X_va_dense = np.nan_to_num(X_val.to_numpy(dtype=np.float32), nan=0.0)
        X_te_dense = np.nan_to_num(X_te.to_numpy(dtype=np.float32), nan=0.0)

        X_tr_scaled = scaler.fit_transform(X_tr_dense)
        X_va_scaled = scaler.transform(X_va_dense)
        X_te_scaled = scaler.transform(X_te_dense)

        model = LogisticRegression(
            penalty="l2", C=0.1, solver="lbfgs", class_weight="balanced", max_iter=500, random_state=self.seed, n_jobs=-1
        )
        model.fit(X_tr_scaled, y_tr)
        train_probs = model.predict_proba(X_tr_scaled)[:, 1]
        val_probs = model.predict_proba(X_va_scaled)[:, 1]
        test_probs = model.predict_proba(X_te_scaled)[:, 1]
        return model, train_probs, val_probs, test_probs

    def train_stageA_xgboost(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray, sub_tr: np.ndarray, X_val: pd.DataFrame, X_te: pd.DataFrame
    ) -> Tuple[xgb.XGBClassifier, np.ndarray, np.ndarray, np.ndarray]:
        """Train Stage A XGBoost Classifier with GroupKFold hyperparameter tuning."""
        log.info("Stage A: Training XGBoost with GroupKFold search & scale_pos_weight...")
        n_pos = np.sum(y_tr == 1)
        n_neg = np.sum(y_tr == 0)
        scale_pos_weight = float(n_neg / max(1, n_pos))

        param_grid = [
            {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 300},
            {"max_depth": 6, "learning_rate": 0.03, "n_estimators": 350},
        ]
        gkf = GroupKFold(n_splits=3)
        best_score = -1.0
        best_params = param_grid[0]

        for params in param_grid:
            scores = []
            for tr_idx, val_idx in gkf.split(X_tr, y_tr, groups=sub_tr):
                clf = xgb.XGBClassifier(
                    **params,
                    scale_pos_weight=scale_pos_weight,
                    random_state=self.seed,
                    n_jobs=-1,
                    eval_metric="logloss",
                )
                clf.fit(X_tr.iloc[tr_idx], y_tr[tr_idx])
                preds = clf.predict(X_tr.iloc[val_idx])
                scores.append(np.mean(preds == y_tr[val_idx]))
            mean_s = float(np.mean(scores))
            if mean_s > best_score:
                best_score = mean_s
                best_params = params

        log.info("Best Stage A XGBoost hyperparams: %s (CV accuracy: %.4f)", best_params, best_score)
        best_clf = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=self.seed,
            n_jobs=-1,
            eval_metric="logloss",
        )
        best_clf.fit(X_tr, y_tr)
        train_probs = best_clf.predict_proba(X_tr)[:, 1]
        val_probs = best_clf.predict_proba(X_val)[:, 1]
        test_probs = best_clf.predict_proba(X_te)[:, 1]
        return best_clf, train_probs, val_probs, test_probs

    def train_stageA_lightgbm(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray, sub_tr: np.ndarray, X_val: pd.DataFrame, X_te: pd.DataFrame
    ) -> Tuple[lgb.LGBMClassifier, np.ndarray, np.ndarray, np.ndarray]:
        """Train Stage A LightGBM Classifier with GroupKFold hyperparameter tuning."""
        log.info("Stage A: Training LightGBM with GroupKFold search & class_weight='balanced'...")
        param_grid = [
            {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300},
            {"num_leaves": 63, "learning_rate": 0.03, "n_estimators": 350},
        ]
        gkf = GroupKFold(n_splits=3)
        best_score = -1.0
        best_params = param_grid[0]

        for params in param_grid:
            scores = []
            for tr_idx, val_idx in gkf.split(X_tr, y_tr, groups=sub_tr):
                clf = lgb.LGBMClassifier(
                    **params,
                    class_weight="balanced",
                    random_state=self.seed,
                    n_jobs=-1,
                    verbose=-1,
                )
                clf.fit(X_tr.iloc[tr_idx], y_tr[tr_idx])
                preds = clf.predict(X_tr.iloc[val_idx])
                scores.append(np.mean(preds == y_tr[val_idx]))
            mean_s = float(np.mean(scores))
            if mean_s > best_score:
                best_score = mean_s
                best_params = params

        log.info("Best Stage A LightGBM hyperparams: %s (CV accuracy: %.4f)", best_params, best_score)
        best_clf = lgb.LGBMClassifier(
            **best_params,
            class_weight="balanced",
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )
        best_clf.fit(X_tr, y_tr)
        train_probs = best_clf.predict_proba(X_tr)[:, 1]
        val_probs = best_clf.predict_proba(X_val)[:, 1]
        test_probs = best_clf.predict_proba(X_te)[:, 1]
        return best_clf, train_probs, val_probs, test_probs

    def calibrate_stageA_predictions(
        self, y_val: np.ndarray, val_probs: np.ndarray, test_probs: np.ndarray
    ) -> Tuple[IsotonicRegression, np.ndarray, np.ndarray]:
        """Fit Isotonic Regression on Stage A validation predictions."""
        log.info("Stage A: Fitting Isotonic Regression calibration on validation predictions...")
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(val_probs, y_val)
        val_p_cal = iso.transform(val_probs)
        test_p_cal = iso.transform(test_probs)
        return iso, val_p_cal, test_p_cal

    # ── STAGE B: REGRESSION WITHIN SHORT BUCKET ─────────────────────────────

    def train_stageB_xgboost(
        self,
        X_tr_short: pd.DataFrame,
        y_tr_short: np.ndarray,
        sub_tr_short: np.ndarray,
        X_val_short: pd.DataFrame,
        X_te: pd.DataFrame,
    ) -> Tuple[xgb.XGBRegressor, np.ndarray, np.ndarray]:
        """Train Stage B XGBoost Regressor on short stay bucket."""
        log.info("Stage B: Training XGBoost Regressor on short stay bucket (N=%d)...", len(y_tr_short))
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=self.seed,
            n_jobs=-1,
        )
        model.fit(X_tr_short, y_tr_short)
        val_preds = model.predict(X_val_short)
        test_preds = model.predict(X_te)
        return model, val_preds, test_preds

    def train_stageB_lightgbm(
        self,
        X_tr_short: pd.DataFrame,
        y_tr_short: np.ndarray,
        sub_tr_short: np.ndarray,
        X_val_short: pd.DataFrame,
        X_te: pd.DataFrame,
    ) -> Tuple[lgb.LGBMRegressor, np.ndarray, np.ndarray]:
        """Train Stage B LightGBM Regressor on short stay bucket."""
        log.info("Stage B: Training LightGBM Regressor on short stay bucket (N=%d)...", len(y_tr_short))
        model = lgb.LGBMRegressor(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression",
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_tr_short, y_tr_short)
        val_preds = model.predict(X_val_short)
        test_preds = model.predict(X_te)
        return model, val_preds, test_preds

    def save_artifacts(
        self,
        stageA_xgb: xgb.XGBClassifier,
        stageA_lgb: lgb.LGBMClassifier,
        stageA_calib: IsotonicRegression,
        stageB_xgb: xgb.XGBRegressor,
        stageB_lgb: lgb.LGBMRegressor,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Save pickled model artifacts."""
        output_dir = output_dir or Path("models")
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = "icu_los" if self.target_name == "icu_los" else "los"
        saved = {}

        paths = {
            f"{prefix}_stageA_classifier_xgboost.pkl": stageA_xgb,
            f"{prefix}_stageA_classifier_lightgbm.pkl": stageA_lgb,
            f"{prefix}_stageA_calibrated.pkl": stageA_calib,
            f"{prefix}_stageB_regressor_xgboost.pkl": stageB_xgb,
            f"{prefix}_stageB_regressor_lightgbm.pkl": stageB_lgb,
        }

        for filename, model_obj in paths.items():
            p = output_dir / filename
            with open(p, "wb") as fh:
                pickle.dump(model_obj, fh)
            saved[filename] = p
            log.info("Saved %s model pickle → %s", prefix.upper(), p)

        return saved

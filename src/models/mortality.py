"""
src/models/mortality.py
────────────────────────
Model training, hyperparameter optimization with GroupKFold, isotonic calibration, and pickling.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from src.features.leakage_filters import (
    WINDOWED_LAB_FEATURES,
    MORTALITY_EXCLUDE_RUN_B,
    MORTALITY_EXCLUDE_RUN_C,
    apply_exclusions,
)
from src.utils.config import CFG
from src.utils.logger import get_logger

log = get_logger(__name__)

# Direct outcome-adjacent exclusions (Run A)
OUTCOME_ADJACENT_ONLY = [
    "deathtime", "dischtime", "discharge_location", "los_days", "los_hours", "dod",
] + WINDOWED_LAB_FEATURES


class MortalityModelPipeline:
    """End-to-end model trainer for in-hospital mortality prediction."""

    def __init__(self, data_path: Optional[Path] = None, split_path: Optional[Path] = None) -> None:
        self.cfg = CFG
        self.data_path = data_path or Path(self.cfg.resolve(self.cfg.paths.processed)) / "admission_level_selected.parquet"
        self.split_path = split_path or Path(self.cfg.resolve(self.cfg.paths.processed)) / "patient_split.parquet"
        self.target_col = "hospital_expire_flag"
        self.seed = self.cfg.random_seed

    def prepare_datasets(
        self, run_type: str = "B"
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
        List[str]
    ]:
        """
        Load, split, and clean datasets for Run A, Run B, or Run C.

        Parameters
        ----------
        run_type : str
            'A' (direct outcome exclusions), 'B' (full exclusions), or 'C' (strict 24h observation window).

        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test, subjects_train, subjects_val, subjects_test, feature_names
        """
        log.info("Loading dataset from %s...", self.data_path.name)
        df = pd.read_parquet(self.data_path)
        log.info("Loading patient split from %s...", self.split_path.name)
        splits = pd.read_parquet(self.split_path)

        # Drop rows with null target
        df = df[df[self.target_col].notna()].copy()
        df[self.target_col] = df[self.target_col].astype(int)
        log.info("Cohort size after dropping missing targets: %d admissions", len(df))

        # Join split keyed on subject_id
        if "split" in df.columns:
            df = df.drop(columns=["split"])
        df = df.merge(splits, on="subject_id", how="left")

        # Zero subject_id overlap verification
        train_sub = set(df[df["split"] == "train"]["subject_id"])
        val_sub = set(df[df["split"] == "val"]["subject_id"])
        test_sub = set(df[df["split"] == "test"]["subject_id"])

        assert len(train_sub & val_sub) == 0, "Leakage error: train & val overlap!"
        assert len(train_sub & test_sub) == 0, "Leakage error: train & test overlap!"
        assert len(val_sub & test_sub) == 0, "Leakage error: val & test overlap!"
        log.info("PASSED ZERO-OVERLAP ASSERTION across splits.")

        # Apply leakage exclusions based on run_type
        rt = run_type.upper()
        if rt == "A":
            exclusion_list = OUTCOME_ADJACENT_ONLY
        elif rt == "C":
            exclusion_list = MORTALITY_EXCLUDE_RUN_C
        else:
            exclusion_list = MORTALITY_EXCLUDE_RUN_B

        log.info("Applying Run %s Exclusion Protocol...", rt)
        df_filtered = apply_exclusions(df, exclusion_list, verbose=True)

        # Exclude metadata, identifiers, raw text, and timestamp columns from X
        non_features = [
            "subject_id", "hadm_id", "note_id", "admit_provider_id", "split", self.target_col,
            "text_clean", "text_tfidf_ready",
        ]
        
        # Drop any datetime columns and high-cardinality text/ID features
        candidate_cols = [c for c in df_filtered.columns if c not in non_features]
        feature_cols = []
        for col in candidate_cols:
            dtype_str = str(df_filtered[col].dtype)
            if "datetime" in dtype_str:
                continue
            if (df_filtered[col].dtype == object or dtype_str == "category") and df_filtered[col].nunique(dropna=False) > 100:
                log.info("Excluding high-cardinality categorical feature '%s' (%d categories)", col, df_filtered[col].nunique(dropna=False))
                continue
            feature_cols.append(col)

        log.info("Selected %d tabular predictor features for model training", len(feature_cols))

        # One-hot encode low-cardinality categoricals safely
        X_all = pd.get_dummies(df_filtered[feature_cols], drop_first=True)
        # Convert boolean columns to int8
        bool_cols = X_all.select_dtypes(include=["bool"]).columns
        X_all[bool_cols] = X_all[bool_cols].astype(np.int8)

        train_mask = df_filtered["split"] == "train"
        val_mask = df_filtered["split"] == "val"
        test_mask = df_filtered["split"] == "test"

        X_train, y_train = X_all[train_mask].copy(), df_filtered.loc[train_mask, self.target_col].to_numpy()
        X_val, y_val = X_all[val_mask].copy(), df_filtered.loc[val_mask, self.target_col].to_numpy()
        X_test, y_test = X_all[test_mask].copy(), df_filtered.loc[test_mask, self.target_col].to_numpy()

        sub_train = df_filtered.loc[train_mask, "subject_id"].to_numpy()
        sub_val = df_filtered.loc[val_mask, "subject_id"].to_numpy()
        sub_test = df_filtered.loc[test_mask, "subject_id"].to_numpy()

        # Print prevalence per split
        log.info("Prevalence (Positive Rate) by Split:")
        log.info("  Train: %d / %d (%.2f%%)", y_train.sum(), len(y_train), 100 * y_train.mean())
        log.info("  Val  : %d / %d (%.2f%%)", y_val.sum(), len(y_val), 100 * y_val.mean())
        log.info("  Test : %d / %d (%.2f%%)", y_test.sum(), len(y_test), 100 * y_test.mean())

        return X_train, X_val, X_test, y_train, y_val, y_test, sub_train, sub_val, sub_test, list(X_all.columns)

    def train_logistic_regression(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[LogisticRegression, np.ndarray, np.ndarray]:
        """Train L2-regularized Logistic Regression baseline with fast scaling."""
        log.info("Training Logistic Regression (L2, class_weight='balanced')...")
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_tr_dense = np.nan_to_num(X_train.to_numpy(dtype=np.float32), nan=0.0)
        X_va_dense = np.nan_to_num(X_val.to_numpy(dtype=np.float32), nan=0.0)
        X_te_dense = np.nan_to_num(X_test.to_numpy(dtype=np.float32), nan=0.0)

        X_train_imp = scaler.fit_transform(X_tr_dense)
        X_val_imp = scaler.transform(X_va_dense)
        X_test_imp = scaler.transform(X_te_dense)

        clf = LogisticRegression(
            penalty="l2",
            C=0.1,
            solver="lbfgs",
            max_iter=200,
            class_weight="balanced",
            random_state=self.seed,
            n_jobs=-1,
        )
        clf.fit(X_train_imp, y_train)

        val_probs = clf.predict_proba(X_val_imp)[:, 1]
        test_probs = clf.predict_proba(X_test_imp)[:, 1]
        return clf, val_probs, test_probs

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sub_train: np.ndarray,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Tuple[xgb.XGBClassifier, np.ndarray, np.ndarray]:
        """Train XGBoost with GroupKFold search and scale_pos_weight."""
        log.info("Training XGBoost with GroupKFold search & scale_pos_weight...")
        scale_pos_weight = float((len(y_train) - y_train.sum()) / max(1, y_train.sum()))
        log.info("XGBoost scale_pos_weight: %.2f", scale_pos_weight)

        best_params = {
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

        # Randomized search over parameter grid using GroupKFold
        gkf = GroupKFold(n_splits=3)
        param_grid = [
            {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 250, "min_child_weight": 5},
            {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 300, "min_child_weight": 3},
            {"max_depth": 6, "learning_rate": 0.03, "n_estimators": 350, "min_child_weight": 3},
        ]

        best_cv_score = -1.0
        for params in param_grid:
            cv_scores = []
            for tr_idx, val_idx in gkf.split(X_train, y_train, groups=sub_train):
                X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

                model = xgb.XGBClassifier(
                    **params,
                    scale_pos_weight=scale_pos_weight,
                    random_state=self.seed,
                    n_jobs=-1,
                    eval_metric="logloss",
                )
                model.fit(X_tr, y_tr)
                score = float(model.score(X_va, y_va))
                cv_scores.append(score)

            mean_score = np.mean(cv_scores)
            if mean_score > best_cv_score:
                best_cv_score = mean_score
                best_params.update(params)

        log.info("Best XGBoost hyperparams: %s (CV accuracy: %.4f)", best_params, best_cv_score)

        final_model = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=self.seed,
            n_jobs=-1,
            eval_metric="logloss",
        )
        final_model.fit(X_train, y_train)

        val_probs = final_model.predict_proba(X_val)[:, 1]
        test_probs = final_model.predict_proba(X_test)[:, 1]
        return final_model, val_probs, test_probs

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sub_train: np.ndarray,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Tuple[lgb.LGBMClassifier, np.ndarray, np.ndarray]:
        """Train LightGBM with GroupKFold search and class_weight='balanced'."""
        log.info("Training LightGBM with GroupKFold search & class_weight='balanced'...")

        best_params = {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

        gkf = GroupKFold(n_splits=3)
        param_grid = [
            {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 250, "min_child_samples": 20},
            {"num_leaves": 45, "learning_rate": 0.03, "n_estimators": 300, "min_child_samples": 30},
            {"num_leaves": 63, "learning_rate": 0.03, "n_estimators": 350, "min_child_samples": 50},
        ]

        best_cv_score = -1.0
        for params in param_grid:
            cv_scores = []
            for tr_idx, val_idx in gkf.split(X_train, y_train, groups=sub_train):
                X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

                model = lgb.LGBMClassifier(
                    **params,
                    class_weight="balanced",
                    random_state=self.seed,
                    n_jobs=-1,
                    verbose=-1,
                )
                model.fit(X_tr, y_tr)
                score = float(model.score(X_va, y_va))
                cv_scores.append(score)

            mean_score = np.mean(cv_scores)
            if mean_score > best_cv_score:
                best_cv_score = mean_score
                best_params.update(params)

        log.info("Best LightGBM hyperparams: %s (CV accuracy: %.4f)", best_params, best_cv_score)

        final_model = lgb.LGBMClassifier(
            **best_params,
            class_weight="balanced",
            random_state=self.seed,
            n_jobs=-1,
            verbose=-1,
        )
        final_model.fit(X_train, y_train)

        val_probs = final_model.predict_proba(X_val)[:, 1]
        test_probs = final_model.predict_proba(X_test)[:, 1]
        return final_model, val_probs, test_probs

    def calibrate_predictions(
        self, y_val: np.ndarray, val_probs: np.ndarray, test_probs: np.ndarray
    ) -> Tuple[IsotonicRegression, np.ndarray, np.ndarray]:
        """Fit Isotonic Regression on validation predictions and calibrate val & test probabilities."""
        log.info("Fitting Isotonic Regression calibration on validation predictions...")
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_probs, y_val)

        calibrated_val_probs = calibrator.transform(val_probs)
        calibrated_test_probs = calibrator.transform(test_probs)
        return calibrator, calibrated_val_probs, calibrated_test_probs

    def save_models(
        self,
        logreg: LogisticRegression,
        xgb_model: xgb.XGBClassifier,
        lgb_model: lgb.LGBMClassifier,
        calibrator: IsotonicRegression,
        models_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Pickle models to disk."""
        models_dir = models_dir or Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}
        items = [
            ("logreg_mortality.pkl", logreg),
            ("xgboost_mortality.pkl", xgb_model),
            ("lightgbm_mortality.pkl", lgb_model),
            ("calibrated_mortality.pkl", calibrator),
        ]

        for fname, model_obj in items:
            p = models_dir / fname
            with open(p, "wb") as fh:
                pickle.dump(model_obj, fh)
            saved_paths[fname] = p
            log.info("Saved model pickle → %s", p)

        return saved_paths

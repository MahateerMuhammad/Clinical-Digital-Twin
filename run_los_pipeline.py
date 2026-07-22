"""
run_los_pipeline.py
───────────────────
Phase 4: Two-Stage Length of Stay (LOS) Prediction & Evaluation Pipeline.
Covers both Hospital LOS (los_days) and ICU LOS (icu_los_days) as parallel target pairs.
Enforces strict 24-hour observation window discipline, GroupKFold patient-level splitting,
Stage A Isotonic Calibration, Stage B Short-Bucket Regression, and SHAP explainability.
"""

from __future__ import annotations

import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.evaluation import (
    evaluate_binary_predictions,
    evaluate_regression_predictions,
    export_los_two_stage_markdown,
    find_optimal_threshold,
)
from src.models.los import LengthOfStayModelPipeline
from src.utils.config import CFG
from src.utils.logger import get_logger
from src.visualization.model_plots import (
    generate_shap_plots,
    plot_calibration_curves,
    plot_roc_pr_curves,
)

log = get_logger(__name__)


def run_target_pipeline(target_name: str = "hospital_los"):
    """Run two-stage pipeline for target 'hospital_los' or 'icu_los'."""
    display_title = "HOSPITAL LENGTH OF STAY (los_days)" if target_name == "hospital_los" else "ICU LENGTH OF STAY (icu_los_days)"
    print("\n" + "=" * 65)
    print(f"   TWO-STAGE PIPELINE: {display_title}")
    print("=" * 65)

    pipeline = LengthOfStayModelPipeline(target_name=target_name)

    (
        X_tr, X_val, X_te,
        y_tr_cls, y_val_cls, y_te_cls,
        y_tr_reg, y_val_reg, y_te_reg,
        sub_tr, sub_val, sub_te,
        p75_threshold, feature_names, df_filtered
    ) = pipeline.prepare_datasets()

    target_label = "Hospital LOS" if target_name == "hospital_los" else "ICU LOS"

    # ── STAGE A: CLASSIFICATION (Short vs. Long Stay) ─────────────────────────
    print(f"\n[STEP 1] Stage A — Classifying Short vs. Long Stay (Threshold = {p75_threshold:.2f} days)...")

    logreg_a, tr_p_logreg_a, val_p_logreg_a, test_p_logreg_a = pipeline.train_stageA_logistic_regression(X_tr, y_tr_cls, X_val, X_te)
    xgb_a, tr_p_xgb_a, val_p_xgb_a, test_p_xgb_a = pipeline.train_stageA_xgboost(X_tr, y_tr_cls, sub_tr, X_val, X_te)
    lgb_a, tr_p_lgb_a, val_p_lgb_a, test_p_lgb_a = pipeline.train_stageA_lightgbm(X_tr, y_tr_cls, sub_tr, X_val, X_te)

    thresh_logreg = find_optimal_threshold(y_val_cls, val_p_logreg_a, target_recall=0.80)
    thresh_xgb = find_optimal_threshold(y_val_cls, val_p_xgb_a, target_recall=0.80)
    thresh_lgb = find_optimal_threshold(y_val_cls, val_p_lgb_a, target_recall=0.80)

    res_logreg = evaluate_binary_predictions(y_te_cls, test_p_logreg_a, threshold=thresh_logreg, model_name="Logistic Regression", run_name="Stage A (Admission-Time)")
    res_xgb = evaluate_binary_predictions(y_te_cls, test_p_xgb_a, threshold=thresh_xgb, model_name="XGBoost", run_name="Stage A (Admission-Time)")
    res_lgb = evaluate_binary_predictions(y_te_cls, test_p_lgb_a, threshold=thresh_lgb, model_name="LightGBM", run_name="Stage A (Admission-Time)")

    res_logreg["target_name"] = target_label
    res_xgb["target_name"] = target_label
    res_lgb["target_name"] = target_label

    # Fit Isotonic Calibration on winning LightGBM model
    calibrator, val_p_cal, test_p_cal = pipeline.calibrate_stageA_predictions(y_val_cls, val_p_lgb_a, test_p_lgb_a)
    cal_thresh = find_optimal_threshold(y_val_cls, val_p_cal, target_recall=0.80)
    res_cal = evaluate_binary_predictions(y_te_cls, test_p_cal, threshold=cal_thresh, model_name="LightGBM (Calibrated)", run_name="Stage A (Admission-Time)")
    res_cal["target_name"] = target_label

    print(f"  Stage A Winning Model: LightGBM (AUROC: {res_lgb['auroc']:.4f}, AUPRC: {res_lgb['auprc']:.4f})")

    # ── STAGE B: REGRESSION WITHIN SHORT BUCKET ──────────────────────────────
    print(f"\n[STEP 2] Stage B — Training Regressors on Short Bucket (stay <= {p75_threshold:.2f} days)...")

    train_short_mask = (y_tr_reg <= p75_threshold)
    val_short_mask = (y_val_reg <= p75_threshold)
    test_actual_short_mask = (y_te_reg <= p75_threshold)

    X_tr_short = X_tr.iloc[train_short_mask]
    y_tr_short = y_tr_reg[train_short_mask]
    sub_tr_short = sub_tr[train_short_mask]

    X_val_short = X_val.iloc[val_short_mask]
    y_val_short = y_val_reg[val_short_mask]

    xgb_b, val_pred_xgb_b, test_pred_xgb_b = pipeline.train_stageB_xgboost(X_tr_short, y_tr_short, sub_tr_short, X_val_short, X_te)
    lgb_b, val_pred_lgb_b, test_pred_lgb_b = pipeline.train_stageB_lightgbm(X_tr_short, y_tr_short, sub_tr_short, X_val_short, X_te)

    # Deployment scenario: Stage A predicted short stay (is_long_stay == 0)
    pred_long_stageA = (test_p_cal >= cal_thresh).astype(int)
    test_pred_short_mask = (pred_long_stageA == 0)

    # Evaluate Stage B Regressors on Test Set under both protocols:
    # 1. Primary Deployment Scenario (Predicted Short Bucket)
    res_reg_xgb_deploy = evaluate_regression_predictions(
        y_true=y_te_reg[test_pred_short_mask],
        y_pred=test_pred_xgb_b[test_pred_short_mask],
        model_name="XGBoost Regressor",
        evaluation_scope="Predicted Short Bucket (Deployment Primary)",
        target_name=target_label,
    )
    res_reg_lgb_deploy = evaluate_regression_predictions(
        y_true=y_te_reg[test_pred_short_mask],
        y_pred=test_pred_lgb_b[test_pred_short_mask],
        model_name="LightGBM Regressor",
        evaluation_scope="Predicted Short Bucket (Deployment Primary)",
        target_name=target_label,
    )

    # 2. Optimistic Upper Bound (Actual Short Bucket)
    res_reg_xgb_opt = evaluate_regression_predictions(
        y_true=y_te_reg[test_actual_short_mask],
        y_pred=test_pred_xgb_b[test_actual_short_mask],
        model_name="XGBoost Regressor",
        evaluation_scope="Actual Short Bucket (Optimistic Upper Bound)",
        target_name=target_label,
    )
    res_reg_lgb_opt = evaluate_regression_predictions(
        y_true=y_te_reg[test_actual_short_mask],
        y_pred=test_pred_lgb_b[test_actual_short_mask],
        model_name="LightGBM Regressor",
        evaluation_scope="Actual Short Bucket (Optimistic Upper Bound)",
        target_name=target_label,
    )

    # Save pickles
    pipeline.save_artifacts(
        stageA_xgb=xgb_a,
        stageA_lgb=lgb_a,
        stageA_calib=calibrator,
        stageB_xgb=xgb_b,
        stageB_lgb=lgb_b,
    )

    # Visualizations & SHAP
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    prefix = "icu_los" if target_name == "icu_los" else "los"

    plot_roc_pr_curves(
        y_true=y_te_cls,
        model_predictions={
            f"LightGBM [AUC={res_lgb['auroc']:.3f}]": test_p_lgb_a,
            f"XGBoost [AUC={res_xgb['auroc']:.3f}]": test_p_xgb_a,
            f"Logistic Regression [AUC={res_logreg['auroc']:.3f}]": test_p_logreg_a,
        },
        base_rate=float(np.mean(y_te_cls)),
        output_path=figures_dir / f"{prefix}_stageA_roc_pr_curves.png",
    )

    plot_calibration_curves(
        y_true=y_te_cls,
        y_prob_uncalibrated=test_p_lgb_a,
        y_prob_calibrated=test_p_cal,
        brier_uncalibrated=res_lgb["brier_score"],
        brier_calibrated=res_cal["brier_score"],
        model_name=f"LightGBM ({target_label} Stage A)",
        output_path=figures_dir / f"{prefix}_stageA_calibration_curves.png",
    )

    try:
        generate_shap_plots(
            model=lgb_a,
            X_train=X_tr,
            X_test=X_te,
            y_test=y_te_cls,
            y_pred_probs=test_p_lgb_a,
            threshold=thresh_lgb,
            output_dir=figures_dir,
            output_prefix=f"{prefix}_stageA",
        )
    except Exception as e:
        log.warning("SHAP generation error for %s: %s", target_name, e)

    cls_results = [res_logreg, res_xgb, res_lgb, res_cal]
    reg_results = [res_reg_lgb_deploy, res_reg_xgb_deploy, res_reg_lgb_opt, res_reg_xgb_opt]

    return p75_threshold, cls_results, reg_results


def run():
    start_time = time.time()
    print("=" * 65)
    print("   PHASE 4: TWO-STAGE LENGTH OF STAY (LOS) PREDICTION PIPELINE")
    print("=" * 65)

    hosp_p75, hosp_cls, hosp_reg = run_target_pipeline(target_name="hospital_los")
    icu_p75, icu_cls, icu_reg = run_target_pipeline(target_name="icu_los")

    all_cls = hosp_cls + icu_cls
    all_reg = hosp_reg + icu_reg

    report_path = export_los_two_stage_markdown(
        classification_results=all_cls,
        regression_results=all_reg,
        hosp_threshold=hosp_p75,
        icu_threshold=icu_p75,
        output_path=Path(CFG.resolve(CFG.paths.reports)) / "tables/los_two_stage_results.md",
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f" PHASE 4 TWO-STAGE LOS PIPELINE COMPLETE IN {elapsed:.2f} SECONDS")
    print(f" Output Report → {report_path}")
    print("=" * 65)


if __name__ == "__main__":
    run()

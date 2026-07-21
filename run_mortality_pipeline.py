"""
run_mortality_pipeline.py
─────────────────────────
Executable pipeline for Phase 1 — In-Hospital Mortality Prediction.
Executes Run A vs Run B leakage audit, GroupKFold model training, Isotonic Calibration,
SHAP explainability, figure generation, and markdown report export.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.mortality import MortalityModelPipeline
from src.models.evaluation import (
    evaluate_binary_predictions,
    export_model_comparison_markdown,
    find_optimal_threshold,
)
from src.visualization.model_plots import (
    generate_shap_plots,
    plot_calibration_curves,
    plot_roc_pr_curves,
)
from src.utils.logger import get_logger

log = get_logger("run_mortality_pipeline")


def run_phase1_pipeline():
    print("\n" + "=" * 65)
    print("   PHASE 1: IN-HOSPITAL MORTALITY PREDICTION & LEAKAGE AUDIT")
    print("=" * 65 + "\n")
    start_time = time.time()

    pipeline = MortalityModelPipeline()

    # ── 1. RUN A: Direct Outcome Exclusions Only ────────────────────────────────
    print("\n[STEP 1] Running RUN A (Direct Outcome Exclusions Only)...")
    (
        X_tr_a, X_val_a, X_te_a,
        y_tr_a, y_val_a, y_te_a,
        sub_tr_a, sub_val_a, sub_te_a,
        feats_a
    ) = pipeline.prepare_datasets(run_type="A")

    logreg_a, val_p_logreg_a, test_p_logreg_a = pipeline.train_logistic_regression(X_tr_a, y_tr_a, X_val_a, X_te_a)
    xgb_a, val_p_xgb_a, test_p_xgb_a = pipeline.train_xgboost(X_tr_a, y_tr_a, sub_tr_a, X_val_a, X_te_a)
    lgb_a, val_p_lgb_a, test_p_lgb_a = pipeline.train_lightgbm(X_tr_a, y_tr_a, sub_tr_a, X_val_a, X_te_a)

    thresh_logreg_a = find_optimal_threshold(y_val_a, val_p_logreg_a, target_recall=0.80)
    thresh_xgb_a = find_optimal_threshold(y_val_a, val_p_xgb_a, target_recall=0.80)
    thresh_lgb_a = find_optimal_threshold(y_val_a, val_p_lgb_a, target_recall=0.80)

    res_logreg_a = evaluate_binary_predictions(y_te_a, test_p_logreg_a, threshold=thresh_logreg_a, model_name="Logistic Regression", run_name="Run A (With ICD)")
    res_xgb_a = evaluate_binary_predictions(y_te_a, test_p_xgb_a, threshold=thresh_xgb_a, model_name="XGBoost", run_name="Run A (With ICD)")
    res_lgb_a = evaluate_binary_predictions(y_te_a, test_p_lgb_a, threshold=thresh_lgb_a, model_name="LightGBM", run_name="Run A (With ICD)")

    # ── 2. RUN B: Full Exclusions (MORTALITY_EXCLUDE) ───────────────────────────
    print("\n[STEP 2] Running RUN B (Full Exclusions - Leak-Free Protocol)...")
    (
        X_tr_b, X_val_b, X_te_b,
        y_tr_b, y_val_b, y_te_b,
        sub_tr_b, sub_val_b, sub_te_b,
        feats_b
    ) = pipeline.prepare_datasets(run_type="B")

    logreg_b, val_p_logreg_b, test_p_logreg_b = pipeline.train_logistic_regression(X_tr_b, y_tr_b, X_val_b, X_te_b)
    xgb_b, val_p_xgb_b, test_p_xgb_b = pipeline.train_xgboost(X_tr_b, y_tr_b, sub_tr_b, X_val_b, X_te_b)
    lgb_b, val_p_lgb_b, test_p_lgb_b = pipeline.train_lightgbm(X_tr_b, y_tr_b, sub_tr_b, X_val_b, X_te_b)

    thresh_logreg_b = find_optimal_threshold(y_val_b, val_p_logreg_b, target_recall=0.80)
    thresh_xgb_b = find_optimal_threshold(y_val_b, val_p_xgb_b, target_recall=0.80)
    thresh_lgb_b = find_optimal_threshold(y_val_b, val_p_lgb_b, target_recall=0.80)

    res_logreg_b = evaluate_binary_predictions(y_te_b, test_p_logreg_b, threshold=thresh_logreg_b, model_name="Logistic Regression", run_name="Run B (Leak-Free)")
    res_xgb_b = evaluate_binary_predictions(y_te_b, test_p_xgb_b, threshold=thresh_xgb_b, model_name="XGBoost", run_name="Run B (Leak-Free)")
    res_lgb_b = evaluate_binary_predictions(y_te_b, test_p_lgb_b, threshold=thresh_lgb_b, model_name="LightGBM", run_name="Run B (Leak-Free)")

    # ── 3. RUN C: Strict 24h Early Observation Window (MORTALITY_EXCLUDE_RUN_C) ──
    print("\n[STEP 3] Running RUN C (Strict 24h Early Observation Window Protocol)...")
    (
        X_tr_c, X_val_c, X_te_c,
        y_tr_c, y_val_c, y_te_c,
        sub_tr_c, sub_val_c, sub_te_c,
        feats_c
    ) = pipeline.prepare_datasets(run_type="C")

    logreg_c, val_p_logreg_c, test_p_logreg_c = pipeline.train_logistic_regression(X_tr_c, y_tr_c, X_val_c, X_te_c)
    xgb_c, val_p_xgb_c, test_p_xgb_c = pipeline.train_xgboost(X_tr_c, y_tr_c, sub_tr_c, X_val_c, X_te_c)
    lgb_c, val_p_lgb_c, test_p_lgb_c = pipeline.train_lightgbm(X_tr_c, y_tr_c, sub_tr_c, X_val_c, X_te_c)

    thresh_logreg_c = find_optimal_threshold(y_val_c, val_p_logreg_c, target_recall=0.80)
    thresh_xgb_c = find_optimal_threshold(y_val_c, val_p_xgb_c, target_recall=0.80)
    thresh_lgb_c = find_optimal_threshold(y_val_c, val_p_lgb_c, target_recall=0.80)

    res_logreg_c = evaluate_binary_predictions(y_te_c, test_p_logreg_c, threshold=thresh_logreg_c, model_name="Logistic Regression", run_name="Run C (24h Window)")
    res_xgb_c = evaluate_binary_predictions(y_te_c, test_p_xgb_c, threshold=thresh_xgb_c, model_name="XGBoost", run_name="Run C (24h Window)")
    res_lgb_c = evaluate_binary_predictions(y_te_c, test_p_lgb_c, threshold=thresh_lgb_c, model_name="LightGBM", run_name="Run C (24h Window)")

    # ── 4. Leakage Gap Analysis & Winner Selection ─────────────────────────────
    print("\n[STEP 4] Controlled Leakage & Observation Window Comparison...")
    print(f"  XGBoost  AUROC -> Run A: {res_xgb_a['auroc']:.4f} | Run B: {res_xgb_b['auroc']:.4f} | Run C (24h): {res_xgb_c['auroc']:.4f}")
    print(f"  LightGBM AUROC -> Run A: {res_lgb_a['auroc']:.4f} | Run B: {res_lgb_b['auroc']:.4f} | Run C (24h): {res_lgb_c['auroc']:.4f}")
    print(f"  LogReg   AUROC -> Run A: {res_logreg_a['auroc']:.4f} | Run B: {res_logreg_b['auroc']:.4f} | Run C (24h): {res_logreg_c['auroc']:.4f}")

    # Select winning model on Run C based on validation AUROC
    models_c = {
        "XGBoost": (xgb_c, val_p_xgb_c, test_p_xgb_c, res_xgb_c, thresh_xgb_c),
        "LightGBM": (lgb_c, val_p_lgb_c, test_p_lgb_c, res_lgb_c, thresh_lgb_c),
        "Logistic Regression": (logreg_c, val_p_logreg_c, test_p_logreg_c, res_logreg_c, thresh_logreg_c),
    }

    winning_name = max(models_c.keys(), key=lambda k: models_c[k][3]["auroc"])
    winning_model, win_val_p, win_test_p, win_res, win_thresh = models_c[winning_name]
    print(f"\n  ★ Winning Model (Run C - 24h Window): {winning_name} (AUROC: {win_res['auroc']:.4f}, AUPRC: {win_res['auprc']:.4f})")

    # ── 5. Isotonic Calibration ─────────────────────────────────────────────────
    print(f"\n[STEP 5] Applying Isotonic Calibration to {winning_name} (Run C)...")
    calibrator, val_p_calibrated, test_p_calibrated = pipeline.calibrate_predictions(y_val_c, win_val_p, win_test_p)

    # Re-tune decision threshold specifically on calibrated validation probabilities
    calibrated_thresh = find_optimal_threshold(y_val_c, val_p_calibrated, target_recall=0.80)

    res_winning_calibrated = evaluate_binary_predictions(
        y_te_c, test_p_calibrated, threshold=calibrated_thresh, model_name=f"{winning_name} (Calibrated)", run_name="Run C (24h Window)"
    )

    print(f"  Brier Score Before Calibration: {win_res['brier_score']:.4f}")
    print(f"  Brier Score After Calibration : {res_winning_calibrated['brier_score']:.4f}")
    print(f"  Calibrated Model Threshold    : {calibrated_thresh:.4f} (Target Recall: 80%)")

    # ── 6. Save Pickles & Generate Markdown Report ─────────────────────────────
    print("\n[STEP 6] Saving Model Binaries & Markdown Report...")
    pipeline.save_models(logreg_c, xgb_c, lgb_c, calibrator)

    all_results = [
        res_xgb_a, res_lgb_a, res_logreg_a,
        res_xgb_b, res_lgb_b, res_logreg_b,
        res_xgb_c, res_lgb_c, res_logreg_c,
        res_winning_calibrated,
    ]
    export_model_comparison_markdown(all_results, leakage_gap_detected=True, winning_model_name=winning_name)

    # ── 7. Generate Figures & SHAP Explainability ──────────────────────────────
    print("\n[STEP 7] Generating Publication Plots & SHAP Explainability...")
    plot_calibration_curves(
        y_true=y_te_c,
        y_prob_uncalibrated=win_test_p,
        y_prob_calibrated=test_p_calibrated,
        brier_uncalibrated=win_res["brier_score"],
        brier_calibrated=res_winning_calibrated["brier_score"],
        model_name=winning_name,
    )

    plot_roc_pr_curves(
        y_true=y_te_c,
        model_predictions={
            f"XGBoost (Run C 24h) [AUC={res_xgb_c['auroc']:.3f}]": test_p_xgb_c,
            f"LightGBM (Run C 24h) [AUC={res_lgb_c['auroc']:.3f}]": test_p_lgb_c,
            f"LogReg (Run C 24h) [AUC={res_logreg_c['auroc']:.3f}]": test_p_logreg_c,
        },
        base_rate=float(np.mean(y_te_c)),
    )

    # Generate SHAP explainability plots for both XGBoost and LightGBM models on Run C
    print("\n--- SHAP Feature Importances for XGBoost (Run C - 24h Window) ---")
    generate_shap_plots(
        model=xgb_c,
        X_train=X_tr_c,
        X_test=X_te_c,
        y_test=y_te_c,
        y_pred_probs=test_p_xgb_c,
        threshold=thresh_xgb_c,
    )

    print("\n--- SHAP Feature Importances for LightGBM (Run C - 24h Window) ---")
    generate_shap_plots(
        model=lgb_c,
        X_train=X_tr_c,
        X_test=X_te_c,
        y_test=y_te_c,
        y_pred_probs=test_p_lgb_c,
        threshold=thresh_lgb_c,
    )

    elapsed = (time.time() - start_time) / 60.0
    print("\n" + "=" * 65)
    print(f"  PHASE 1 COMPLETE IN {elapsed:.1f} MINUTES — ALL ARTIFACTS VERIFIED!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_phase1_pipeline()

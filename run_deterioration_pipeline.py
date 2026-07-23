"""
run_deterioration_pipeline.py
──────────────────────────────
Executable pipeline script for Phase 5: Clinical Deterioration Prediction.
Trains Logistic Regression, XGBoost, and LightGBM models under GroupKFold CV on subject_id,
computes NEWS2 scores, enforces strict 6-hour prediction window discipline, performs
isotonic calibration, evaluates SHAP explainability, and generates figures, tables, and pickles.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

from src.models.deterioration import DeteriorationModelPipeline
from src.utils.logger import get_logger

log = get_logger("run_deterioration_pipeline")


def main() -> None:
    log.info("Starting Phase 5: Clinical Deterioration Prediction Pipeline...")

    pipeline = DeteriorationModelPipeline(window_hours=6.0)

    # 1. Prepare Datasets & Enforce Window Discipline
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        sub_train, sub_val, sub_test,
        feature_names, leakage_results
    ) = pipeline.prepare_datasets()

    # 2. Worked Example Verification
    # Load raw admission data to display worked example details
    adm_df = pd.read_parquet(pipeline.data_path)
    icu_df = pd.read_parquet(pipeline.icu_path)
    icu_sorted = icu_df.sort_values("intime")
    icu_first = icu_sorted.groupby("hadm_id", as_index=False).first()
    if "intime" in adm_df.columns:
        adm_df = adm_df.drop(columns=["intime", "outtime"], errors="ignore")
    adm_df = adm_df.merge(icu_first[["hadm_id", "stay_id", "intime", "outtime"]], on="hadm_id", how="left")
    adm_df["admittime"] = pd.to_datetime(adm_df["admittime"])
    adm_df["intime"] = pd.to_datetime(adm_df["intime"])
    adm_df["time_to_icu_hrs"] = (adm_df["intime"] - adm_df["admittime"]).dt.total_seconds() / 3600.0
    adm_df[pipeline.target_col] = ((adm_df["has_icu_stay"] == 1) & (adm_df["time_to_icu_hrs"] > 6.0)).astype(int)

    worked_ex = pipeline.get_worked_example(adm_df)
    log.info("WORKED EXAMPLE VERIFICATION (Subject ID: %d, HADM ID: %d):", worked_ex["subject_id"], worked_ex["hadm_id"])
    log.info("  Admission Time : %s", worked_ex["admittime"])
    log.info("  ICU Event Time : %s", worked_ex["event_time"])
    log.info("  6h Cutoff Time : %s", worked_ex["cutoff_time"])
    log.info("  Timetable of Vitals Events:\n%s", worked_ex["events_timetable"].to_string())

    # 3. Train & Evaluate Models
    results_df, predictions = pipeline.train_eval_models(
        X_train, X_val, X_test, y_train, y_val, y_test, sub_train
    )

    # 4. Isotonic Calibration of Winning Model
    calibrated_probs, brier_pre, brier_post = pipeline.calibrate_best_model(
        X_val, y_val, X_test, y_test
    )

    # Update results dataframe with calibrated brier score
    win_mask = results_df["Model"] == pipeline.winning_model_name
    results_df.loc[win_mask, "Brier Score Post-Calib"] = round(float(brier_post), 4)

    # 5. SHAP Explainability Analysis
    shap_values, explanation = pipeline.compute_shap_explainability(X_test, feature_names)

    # 6. Save Artifacts & Figures
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure A: ROC & PR Curves
    plt.figure(figsize=(12, 5))
    
    # ROC Plot
    plt.subplot(1, 2, 1)
    for name, probs in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUROC={roc_auc_score(y_test, probs):.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Clinical Deterioration ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # PR Plot
    plt.subplot(1, 2, 2)
    base_rate = y_test.mean()
    for name, probs in predictions.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        auprc_val = auc(rec, prec)
        plt.plot(rec, prec, label=f"{name} (AUPRC={auprc_val:.3f})")
    plt.axhline(base_rate, color="k", linestyle="--", label=f"Base Rate ({base_rate:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Clinical Deterioration PR Curves")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    roc_pr_path = figures_dir / "deterioration_roc_pr_curves.png"
    plt.savefig(roc_pr_path, dpi=300)
    plt.close()
    log.info("Saved ROC/PR curves figure to %s", roc_pr_path)

    # Figure B: Calibration Curve
    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(7, 6))
    raw_win_probs = predictions[pipeline.winning_model_name]
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, raw_win_probs, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, calibrated_probs, n_bins=10)

    plt.plot(prob_pred_raw, prob_true_raw, "s-", label=f"Uncalibrated (Brier={brier_pre:.4f})")
    plt.plot(prob_pred_cal, prob_true_cal, "o-", label=f"Isotonic Calibrated (Brier={brier_post:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve - {pipeline.winning_model_name}")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    calib_path = figures_dir / "deterioration_calibration.png"
    plt.savefig(calib_path, dpi=300)
    plt.close()
    log.info("Saved calibration curve figure to %s", calib_path)

    # Figure C: SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    sample_X = X_test.head(1000).fillna(0)
    shap.summary_plot(shap_values, sample_X, show=False)
    plt.title(f"SHAP Feature Importance Summary - {pipeline.winning_model_name}")
    plt.tight_layout()
    shap_path = figures_dir / "deterioration_shap_summary.png"
    plt.savefig(shap_path, dpi=300)
    plt.close()
    log.info("Saved SHAP summary plot to %s", shap_path)

    # 7. Save Models and Report Table
    saved_artifacts = pipeline.save_artifacts(results_df=results_df)
    log.info("Phase 5 Pipeline completed successfully! Created %d artifacts.", len(saved_artifacts) + 3)


if __name__ == "__main__":
    main()

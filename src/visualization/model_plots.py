"""
src/visualization/model_plots.py
─────────────────────────────────
Visualization module for calibration curves, ROC/PR curves, and SHAP explainability plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from src.utils.logger import get_logger

log = get_logger(__name__)

# Use standard publication aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


def plot_calibration_curves(
    y_true: np.ndarray,
    y_prob_uncalibrated: np.ndarray,
    y_prob_calibrated: np.ndarray,
    brier_uncalibrated: float,
    brier_calibrated: float,
    model_name: str = "XGBoost",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Plot reliability calibration diagram comparing pre- vs post-isotonic calibration.
    """
    output_path = output_path or Path("reports/figures/mortality_calibration_curve.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, y_prob_uncalibrated, n_bins=10, strategy="uniform")
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_prob_calibrated, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    # Perfectly calibrated reference line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", alpha=0.7)

    ax.plot(
        prob_pred_uncal, prob_true_uncal, "s-", color="#e74c3c", linewidth=2,
        label=f"{model_name} (Uncalibrated) — Brier = {brier_uncalibrated:.4f}"
    )
    ax.plot(
        prob_pred_cal, prob_true_cal, "o-", color="#2ecc71", linewidth=2,
        label=f"{model_name} (Isotonic Calibrated) — Brier = {brier_calibrated:.4f}"
    )

    ax.set_title(f"Reliability Calibration Curve ({model_name} Mortality Model)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives (Observed Frequency)", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    log.info("Saved calibration plot → %s", output_path)
    return output_path


def plot_roc_pr_curves(
    y_true: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
    base_rate: float,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Plot ROC and Precision-Recall curves for all evaluated models.
    """
    output_path = output_path or Path("reports/figures/mortality_roc_pr_curves.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a"]

    # 1. ROC Curves
    ax_roc.plot([0, 1], [0, 1], "k--", label="Chance (AUROC = 0.50)", alpha=0.6)
    for idx, (m_name, probs) in enumerate(model_predictions.items()):
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax_roc.plot(fpr, tpr, label=m_name, color=colors[idx % len(colors)], linewidth=2)

    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate", fontsize=10)
    ax_roc.legend(loc="lower right", frameon=True, fontsize=9)

    # 2. Precision-Recall Curves
    ax_pr.axhline(y=base_rate, color="gray", linestyle="--", label=f"Base Rate ({base_rate:.4f})", alpha=0.7)
    for idx, (m_name, probs) in enumerate(model_predictions.items()):
        prec, rec, _ = precision_recall_curve(y_true, probs)
        ax_pr.plot(rec, prec, label=m_name, color=colors[idx % len(colors)], linewidth=2)

    ax_pr.set_title("Precision-Recall (PR) Curve", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=10)
    ax_pr.set_ylabel("Precision (PPV)", fontsize=10)
    ax_pr.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    log.info("Saved ROC & PR curves plot → %s", output_path)
    return output_path


def generate_shap_plots(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float,
    output_dir: Optional[Path] = None,
    output_prefix: str = "mortality",
) -> Dict[str, Path]:
    """
    Generate SHAP explainability summary plot and individual case waterfall/bar plots.
    """
    output_dir = output_dir or Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Computing SHAP values for model explainability...")
    explainer = shap.TreeExplainer(model)
    
    # Subsample test set if large for fast SHAP computation
    sample_size = min(2000, len(X_test))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=sample_size, replace=False)

    X_sample = X_test.iloc[sample_idx]
    shap_values = explainer(X_sample)

    # If classification output is 2D (binary classes), take positive class [..., 1]
    if len(shap_values.shape) == 3:
        shap_vals_summary = shap_values[:, :, 1]
    else:
        shap_vals_summary = shap_values

    # Compute and print Top-10 mean absolute SHAP feature importances
    mean_abs_shap = np.abs(shap_vals_summary.values).mean(axis=0)
    top_10_idx = np.argsort(mean_abs_shap)[::-1][:10]
    log.info("TOP-10 SHAP FEATURES FOR MODEL:")
    for rank, idx in enumerate(top_10_idx, 1):
        feat_name = X_sample.columns[idx]
        score = mean_abs_shap[idx]
        log.info("  %2d. %-35s (mean |SHAP| = %.4f)", rank, feat_name, score)

    # 1. Global Summary Plot
    summary_path = output_dir / f"{output_prefix}_shap_summary.png"
    plt.figure(figsize=(9, 7), dpi=300)
    shap.summary_plot(shap_vals_summary, X_sample, show=False, max_display=15)
    plt.title(f"SHAP Feature Importance ({output_prefix.capitalize()} Prediction)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Saved SHAP summary plot → %s", summary_path)

    # 2. Individual Case Plots (TP, TN, FP)
    y_pred_binary = (y_pred_probs >= threshold).astype(int)

    tp_idx = np.where((y_test == 1) & (y_pred_binary == 1))[0]
    tn_idx = np.where((y_test == 0) & (y_pred_binary == 0))[0]
    fp_idx = np.where((y_test == 0) & (y_pred_binary == 1))[0]

    case_paths = {"summary": summary_path}
    cases = [
        ("True Positive (Correctly Flagged Mortality Risk)", tp_idx, "mortality_shap_tp.png", "tp"),
        ("True Negative (Correctly Identified Survivor)", tn_idx, "mortality_shap_tn.png", "tn"),
        ("False Positive (False Alarm / High Risk Survivor)", fp_idx, "mortality_shap_fp.png", "fp"),
    ]

    for title, idx_arr, fname, key in cases:
        if len(idx_arr) > 0:
            target_idx = idx_arr[0]
            row_data = X_test.iloc[[target_idx]]
            exp_val = explainer(row_data)

            if len(exp_val.shape) == 3:
                single_shap = exp_val[0, :, 1]
            else:
                single_shap = exp_val[0]

            save_p = output_dir / fname
            fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
            
            # Create a clean horizontal bar plot for individual feature contributions
            top_k = 10
            vals = single_shap.values
            f_names = np.array(single_shap.feature_names)
            feat_vals = row_data.iloc[0].values

            # Sort by absolute SHAP value magnitude
            top_indices = np.argsort(np.abs(vals))[-top_k:]
            
            y_pos = np.arange(len(top_indices))
            bar_colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals[top_indices]]

            labels = [f"{f_names[i]} = {feat_vals[i]}" if not isinstance(feat_vals[i], float) else f"{f_names[i]} = {feat_vals[i]:.2f}" for i in top_indices]

            ax.barh(y_pos, vals[top_indices], color=bar_colors, align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel("SHAP Value (Impact on Mortality Log-Odds)", fontsize=10)
            ax.set_title(f"Individual Case Breakdown: {title}", fontsize=11, fontweight="bold")
            ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)

            plt.tight_layout()
            fig.savefig(save_p, dpi=300, bbox_inches="tight")
            plt.close(fig)

            case_paths[key] = save_p
            log.info("Saved individual SHAP plot (%s) → %s", key, save_p)

    return case_paths

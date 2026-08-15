"""
src/models/evaluation.py
─────────────────────────
Evaluation metrics, threshold selection, calibration scoring, and markdown report generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

log = get_logger(__name__)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.80,
) -> float:
    """
    Find probability decision threshold on validation set aiming for a target recall (default ~80%).

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels.
    y_prob : array-like
        Predicted probabilities.
    target_recall : float
        Target sensitivity/recall level.

    Returns
    -------
    float
        Optimal probability threshold.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Filter thresholds where recall >= target_recall
    valid_idx = np.where(recalls[:-1] >= target_recall)[0]
    if len(valid_idx) > 0:
        # Pick threshold with highest precision among those achieving target recall
        best_idx = valid_idx[np.argmax(precisions[:-1][valid_idx])]
        return float(thresholds[best_idx])

    # Fallback to threshold giving highest F1 score
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: Optional[float] = None,
    target_recall: float = 0.80,
    model_name: str = "Model",
    run_name: str = "Run B",
) -> Dict[str, float]:
    """
    Compute full evaluation metrics on test set.

    Metrics computed:
    - AUROC
    - AUPRC
    - Trivial AUPRC Baseline (Prevalence)
    - Brier Score
    - Threshold
    - F1 Score
    - Precision
    - Recall

    Returns
    -------
    dict
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    base_rate = float(np.mean(y_true))

    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_prob, target_recall=target_recall)

    y_pred = (y_prob >= threshold).astype(int)

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    metrics = {
        "model_name": model_name,
        "run_name": run_name,
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "base_rate_auprc": round(base_rate, 4),
        "brier_score": round(brier, 4),
        "threshold": round(threshold, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
    }

    log.info(
        "[%s - %s] AUROC: %.4f | AUPRC: %.4f (Base: %.4f) | Brier: %.4f | Thresh: %.4f | F1: %.4f | Prec: %.4f | Rec: %.4f",
        model_name, run_name, auroc, auprc, base_rate, brier, threshold, f1, prec, rec,
    )

    return metrics


def export_model_comparison_markdown(
    results: List[Dict[str, float]],
    output_path: Optional[Path] = None,
    leakage_gap_detected: bool = False,
    winning_model_name: str = "XGBoost",
    title: str = "In-Hospital Mortality Prediction — Model Comparison & Leakage Audit",
    primary_protocol: str = "Run C (24h Window)",
) -> Path:
    """
    Export the side-by-side run comparison as a Markdown table.

    Parameters
    ----------
    title : str
        Report heading. Previously hardcoded to the mortality wording, which is
        why the readmission report was published under the mortality title.
    primary_protocol : str
        Which run is reported as the headline result. Defaults to the strict
        24-hour observation window (Run C), which is what `models/best_models/
        README.md`, the Phase 6 sequence comparison and the Phase 9 risk
        stratification all use. The previous hardcoded "Run B" made this table
        the only artifact in the project claiming a different primary figure.
    """
    output_path = output_path or Path("reports/tables/mortality_model_comparison.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_res = pd.DataFrame(results)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
        fh.write("## 1. Executive Summary & Leakage Protocol Audit\n\n")
        if leakage_gap_detected:
            fh.write(
                "> [!WARNING]\n"
                "> **Leakage Audit Flagged:** Run A (including diagnosis/Charlson ICD codes) achieved an AUROC "
                "> significantly higher (>0.05 gap) than the leak-free runs. In MIMIC-IV, ICD codes are assigned "
                "> post-hoc at discharge, so Run A is reported for audit purposes only and must not be quoted "
                "> as a performance result.\n\n"
            )
        else:
            fh.write(
                "> [!NOTE]\n"
                "> **Leakage Audit Passed:** the leak-free runs explicitly exclude post-hoc ICD diagnosis codes "
                "> (`cci_*`, `dx_*`) to guarantee leak-free evaluation on baseline EHR features.\n\n"
            )

        fh.write(
            f"**Headline result — {primary_protocol}:** `{winning_model_name}`\n\n"
            "> Run A retains post-hoc ICD codes and is an upper bound under leakage, not a result. "
            "Run B removes diagnosis-derived features. "
            "**Run C additionally enforces the strict 24-hour observation window and is the figure "
            "quoted throughout the rest of the project.**\n\n"
        )
        fh.write("## 2. Test Set Performance Comparison Table\n\n")

        # Format markdown table
        fh.write("| Model | Feature Set | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 | Precision | Recall |\n")
        fh.write("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        for r in results:
            fh.write(
                f"| **{r['model_name']}** | {r['run_name']} | **{r['auroc']:.4f}** | **{r['auprc']:.4f}** | "
                f"{r['base_rate_auprc']:.4f} | {r['brier_score']:.4f} | {r['threshold']:.4f} | "
                f"{r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |\n"
            )

        fh.write("\n## 3. Clinical Decision Threshold Rationale\n\n")
        fh.write(
            "The decision threshold was tuned on validation out-of-sample predictions aiming for a **target recall of ~80%** "
            "(high sensitivity) to prioritize flagging deteriorating patients early for clinical intervention.\n"
        )

    log.info("Saved model comparison report → %s", output_path)
    return output_path


def evaluate_regression_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    evaluation_scope: str = "Predicted Short Bucket (Deployment Primary)",
    target_name: str = "Hospital LOS (los_days)",
) -> Dict[str, float]:
    """
    Compute regression metrics (MAE, RMSE, R²) within a restricted short stay bucket.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        log.warning("[%s - %s] Empty evaluation set provided for regression!", model_name, evaluation_scope)
        return {
            "model_name": model_name,
            "target_name": target_name,
            "evaluation_scope": evaluation_scope,
            "n_samples": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
        }

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        "model_name": model_name,
        "target_name": target_name,
        "evaluation_scope": evaluation_scope,
        "n_samples": int(len(y_true)),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }

    log.info(
        "[%s - %s] (%s, N=%d) MAE: %.4f days | RMSE: %.4f days | R²: %.4f",
        model_name, target_name, evaluation_scope, len(y_true), mae, rmse, r2,
    )

    return metrics


def export_los_two_stage_markdown(
    classification_results: List[Dict[str, float]],
    regression_results: List[Dict[str, float]],
    hosp_threshold: float,
    icu_threshold: float,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Export side-by-side Two-Stage Length of Stay (Hospital LOS & ICU LOS) report as Markdown.
    """
    from src.utils.config import CFG
    output_path = output_path or Path(CFG.resolve(CFG.paths.reports)) / "tables/los_two_stage_results.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("# Two-Stage Length of Stay (LOS) Prediction — Model Results & Methodological Audit\n\n")

        fh.write("## 1. Methodological Rationale & Two-Stage Framework\n\n")
        fh.write(
            "> [!NOTE]\n"
            "> **Literature Rationale:** Multiple MIMIC-IV LOS studies found that direct regression using only "
            "> early/admission-time features performs poorly across the full LOS range, because LOS has a long "
            "> right tail (a small number of very long stays) that early features cannot predict well. The consistently "
            "> recommended framework across this literature is: (1) classify short vs. long stay first, (2) only apply "
            "> regression to predict exact duration within the 'short' bucket, and explicitly acknowledge the limitation "
            "> that this framework is not designed to precisely predict long-stay durations.\n\n"
        )
        fh.write(
            f"**Empirical 75th Percentile Thresholds (Training Set Split):**\n"
            f"- **Hospital LOS (`los_days`):** `{hosp_threshold:.2f}` days\n"
            f"- **ICU LOS (`icu_los_days`):** `{icu_threshold:.2f}` days (evaluated on ICU admission cohort)\n\n"
        )

        fh.write("## 2. Stage A — Short vs. Long Stay Classification Performance\n\n")
        fh.write("| Target | Model Name | Run Protocol | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 | Precision | Recall |\n")
        fh.write("|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        for r in classification_results:
            fh.write(
                f"| {r.get('target_name', 'Hospital LOS')} | **{r['model_name']}** | {r['run_name']} | "
                f"**{r['auroc']:.4f}** | **{r['auprc']:.4f}** | {r['base_rate_auprc']:.4f} | "
                f"{r['brier_score']:.4f} | {r['threshold']:.4f} | {r['f1']:.4f} | "
                f"{r['precision']:.4f} | {r['recall']:.4f} |\n"
            )

        fh.write("\n## 3. Stage B — Short-Bucket Duration Regression Performance\n\n")
        fh.write(
            "> [!IMPORTANT]\n"
            "> **Evaluation Scope Discipline:** Stage B regression metrics (MAE, RMSE, R²) are evaluated **strictly within the restricted short-stay bucket** "
            "> (`<= 75th percentile threshold`). Primary deployment metrics reflect performance on the **predicted short bucket** (Stage A classifier output), "
            "> while actual-bucket metrics serve as an optimistic upper bound.\n\n"
        )
        fh.write("| Target | Model Name | Evaluation Protocol (Scope) | Sample Size (N) | MAE (days) | RMSE (days) | R² Score |\n")
        fh.write("|:---|:---|:---|:---:|:---:|:---:|:---:|\n")

        for r in regression_results:
            fh.write(
                f"| {r['target_name']} | **{r['model_name']}** | {r['evaluation_scope']} | "
                f"{r['n_samples']:,} | **{r['mae']:.4f}** | **{r['rmse']:.4f}** | **{r['r2']:.4f}** |\n"
            )

        fh.write("\n## 4. Key Observations & Framework Limitations\n\n")
        fh.write(
            "1. **Right-Tail Isolation:** Classifying long stays in Stage A effectively isolates extreme outliers (>75th percentile), preventing regression skew.\n"
            "2. **Deployment Realism:** Evaluating Stage B on predicted short-stay patients captures real-world error propagation from Stage A classification.\n"
            "3. **Explicit Scope Limitation:** Exact duration predictions are provided ONLY for the short-stay bucket; long-stay cases are flagged for clinical review without artificial exact-day estimates.\n"
        )

    log.info("Saved two-stage LOS comparison report → %s", output_path)
    return output_path


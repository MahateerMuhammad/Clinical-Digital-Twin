"""
create_los_notebook.py
──────────────────────
Script to generate notebooks/10_los_two_stage.ipynb for Phase 4.
"""

import json
from pathlib import Path

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phase 4 — Two-Stage Length of Stay (LOS) Prediction (Hospital & ICU)\n",
                "\n",
                "## Technical & Methodological Rationale\n",
                "\n",
                "Multiple MIMIC-IV LOS studies found that direct regression using only early/admission-time features performs poorly across the full LOS range, because LOS has a long right tail (a small number of very long stays) that early features can't predict well. The consistently recommended framework across this literature is: (1) classify short vs. long stay first, (2) only apply regression to predict exact duration within the \"short\" bucket, and explicitly acknowledge the limitation that this framework is not designed to precisely predict long-stay durations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Add project root to sys.path\n",
                "project_root = Path.cwd()\n",
                "while not (project_root / 'configs' / 'config.yaml').exists():\n",
                "    project_root = project_root.parent\n",
                "sys.path.insert(0, str(project_root))\n",
                "\n",
                "from src.utils.config import CFG\n",
                "from src.utils.logger import get_logger\n",
                "from src.models.los import LengthOfStayModelPipeline\n",
                "from src.models.evaluation import (\n",
                "    evaluate_binary_predictions,\n",
                "    evaluate_regression_predictions,\n",
                "    export_los_two_stage_markdown,\n",
                "    find_optimal_threshold,\n",
                ")\n",
                "from src.visualization.model_plots import (\n",
                "    plot_roc_pr_curves,\n",
                "    plot_calibration_curves,\n",
                "    generate_shap_plots,\n",
                ")\n",
                "\n",
                "log = get_logger('notebook_10_los')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Empirical Threshold Selection (Training Split 75th Percentile)\n",
                "\n",
                "We compute the 75th percentile threshold on the training split to separate Short vs. Long stays dynamically."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Initialize pipelines\n",
                "pipeline_hosp = LengthOfStayModelPipeline(target_name='hospital_los')\n",
                "pipeline_icu = LengthOfStayModelPipeline(target_name='icu_los')\n",
                "\n",
                "(X_tr_h, X_val_h, X_te_h,\n",
                " y_tr_cls_h, y_val_cls_h, y_te_cls_h,\n",
                " y_tr_reg_h, y_val_reg_h, y_te_reg_h,\n",
                " sub_tr_h, sub_val_h, sub_te_h,\n",
                " hosp_p75, feat_h, df_h) = pipeline_hosp.prepare_datasets()\n",
                "\n",
                "(X_tr_i, X_val_i, X_te_i,\n",
                " y_tr_cls_i, y_val_cls_i, y_te_cls_i,\n",
                " y_tr_reg_i, y_val_reg_i, y_te_reg_i,\n",
                " sub_tr_i, sub_val_i, sub_te_i,\n",
                " icu_p75, feat_i, df_i) = pipeline_icu.prepare_datasets()\n",
                "\n",
                "print(f'★ Empirical 75th Percentile Hospital LOS Threshold: {hosp_p75:.2f} days')\n",
                "print(f'★ Empirical 75th Percentile ICU LOS Threshold     : {icu_p75:.2f} days')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Hospital Length of Stay (`los_days`) — Two-Stage Model Training & Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Stage A Classification (Hospital LOS)\n",
                "logreg_a_h, tr_p_lr_h, val_p_lr_h, test_p_lr_h = pipeline_hosp.train_stageA_logistic_regression(X_tr_h, y_tr_cls_h, X_val_h, X_te_h)\n",
                "xgb_a_h, tr_p_xgb_h, val_p_xgb_h, test_p_xgb_h = pipeline_hosp.train_stageA_xgboost(X_tr_h, y_tr_cls_h, sub_tr_h, X_val_h, X_te_h)\n",
                "lgb_a_h, tr_p_lgb_h, val_p_lgb_h, test_p_lgb_h = pipeline_hosp.train_stageA_lightgbm(X_tr_h, y_tr_cls_h, sub_tr_h, X_val_h, X_te_h)\n",
                "\n",
                "t_lr_h = find_optimal_threshold(y_val_cls_h, val_p_lr_h, target_recall=0.80)\n",
                "t_xgb_h = find_optimal_threshold(y_val_cls_h, val_p_xgb_h, target_recall=0.80)\n",
                "t_lgb_h = find_optimal_threshold(y_val_cls_h, val_p_lgb_h, target_recall=0.80)\n",
                "\n",
                "res_lr_h = evaluate_binary_predictions(y_te_cls_h, test_p_lr_h, threshold=t_lr_h, model_name='Logistic Regression', run_name='Stage A')\n",
                "res_xgb_h = evaluate_binary_predictions(y_te_cls_h, test_p_xgb_h, threshold=t_xgb_h, model_name='XGBoost', run_name='Stage A')\n",
                "res_lgb_h = evaluate_binary_predictions(y_te_cls_h, test_p_lgb_h, threshold=t_lgb_h, model_name='LightGBM', run_name='Stage A')\n",
                "\n",
                "# Isotonic Calibration\n",
                "iso_h, val_p_cal_h, test_p_cal_h = pipeline_hosp.calibrate_stageA_predictions(y_val_cls_h, val_p_lgb_h, test_p_lgb_h)\n",
                "t_cal_h = find_optimal_threshold(y_val_cls_h, val_p_cal_h, target_recall=0.80)\n",
                "res_cal_h = evaluate_binary_predictions(y_te_cls_h, test_p_cal_h, threshold=t_cal_h, model_name='LightGBM (Calibrated)', run_name='Stage A')\n",
                "\n",
                "res_lr_h['target_name'] = 'Hospital LOS'\n",
                "res_xgb_h['target_name'] = 'Hospital LOS'\n",
                "res_lgb_h['target_name'] = 'Hospital LOS'\n",
                "res_cal_h['target_name'] = 'Hospital LOS'\n",
                "\n",
                "# Stage B Regression (Hospital LOS)\n",
                "tr_short_m_h = (y_tr_reg_h <= hosp_p75)\n",
                "val_short_m_h = (y_val_reg_h <= hosp_p75)\n",
                "te_act_short_m_h = (y_te_reg_h <= hosp_p75)\n",
                "\n",
                "xgb_b_h, _, te_pred_xgb_b_h = pipeline_hosp.train_stageB_xgboost(X_tr_h.iloc[tr_short_m_h], y_tr_reg_h[tr_short_m_h], sub_tr_h[tr_short_m_h], X_val_h.iloc[val_short_m_h], X_te_h)\n",
                "lgb_b_h, _, te_pred_lgb_b_h = pipeline_hosp.train_stageB_lightgbm(X_tr_h.iloc[tr_short_m_h], y_tr_reg_h[tr_short_m_h], sub_tr_h[tr_short_m_h], X_val_h.iloc[val_short_m_h], X_te_h)\n",
                "\n",
                "# Deployment predicted short bucket\n",
                "pred_long_h = (test_p_cal_h >= t_cal_h).astype(int)\n",
                "te_pred_short_m_h = (pred_long_h == 0)\n",
                "\n",
                "res_reg_lgb_dep_h = evaluate_regression_predictions(y_te_reg_h[te_pred_short_m_h], te_pred_lgb_b_h[te_pred_short_m_h], 'LightGBM Regressor', 'Predicted Short Bucket (Deployment Primary)', 'Hospital LOS')\n",
                "res_reg_xgb_dep_h = evaluate_regression_predictions(y_te_reg_h[te_pred_short_m_h], te_pred_xgb_b_h[te_pred_short_m_h], 'XGBoost Regressor', 'Predicted Short Bucket (Deployment Primary)', 'Hospital LOS')\n",
                "res_reg_lgb_opt_h = evaluate_regression_predictions(y_te_reg_h[te_act_short_m_h], te_pred_lgb_b_h[te_act_short_m_h], 'LightGBM Regressor', 'Actual Short Bucket (Optimistic Upper Bound)', 'Hospital LOS')\n",
                "res_reg_xgb_opt_h = evaluate_regression_predictions(y_te_reg_h[te_act_short_m_h], te_pred_xgb_b_h[te_act_short_m_h], 'XGBoost Regressor', 'Actual Short Bucket (Optimistic Upper Bound)', 'Hospital LOS')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. ICU Length of Stay (`icu_los_days`) — Two-Stage Model Training & Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Stage A Classification (ICU LOS)\n",
                "logreg_a_i, tr_p_lr_i, val_p_lr_i, test_p_lr_i = pipeline_icu.train_stageA_logistic_regression(X_tr_i, y_tr_cls_i, X_val_i, X_te_i)\n",
                "xgb_a_i, tr_p_xgb_i, val_p_xgb_i, test_p_xgb_i = pipeline_icu.train_stageA_xgboost(X_tr_i, y_tr_cls_i, sub_tr_i, X_val_i, X_te_i)\n",
                "lgb_a_i, tr_p_lgb_i, val_p_lgb_i, test_p_lgb_i = pipeline_icu.train_stageA_lightgbm(X_tr_i, y_tr_cls_i, sub_tr_i, X_val_i, X_te_i)\n",
                "\n",
                "t_lr_i = find_optimal_threshold(y_val_cls_i, val_p_lr_i, target_recall=0.80)\n",
                "t_xgb_i = find_optimal_threshold(y_val_cls_i, val_p_xgb_i, target_recall=0.80)\n",
                "t_lgb_i = find_optimal_threshold(y_val_cls_i, val_p_lgb_i, target_recall=0.80)\n",
                "\n",
                "res_lr_i = evaluate_binary_predictions(y_te_cls_i, test_p_lr_i, threshold=t_lr_i, model_name='Logistic Regression', run_name='Stage A')\n",
                "res_xgb_i = evaluate_binary_predictions(y_te_cls_i, test_p_xgb_i, threshold=t_xgb_i, model_name='XGBoost', run_name='Stage A')\n",
                "res_lgb_i = evaluate_binary_predictions(y_te_cls_i, test_p_lgb_i, threshold=t_lgb_i, model_name='LightGBM', run_name='Stage A')\n",
                "\n",
                "# Isotonic Calibration\n",
                "iso_i, val_p_cal_i, test_p_cal_i = pipeline_icu.calibrate_stageA_predictions(y_val_cls_i, val_p_lgb_i, test_p_lgb_i)\n",
                "t_cal_i = find_optimal_threshold(y_val_cls_i, val_p_cal_i, target_recall=0.80)\n",
                "res_cal_i = evaluate_binary_predictions(y_te_cls_i, test_p_cal_i, threshold=t_cal_i, model_name='LightGBM (Calibrated)', run_name='Stage A')\n",
                "\n",
                "res_lr_i['target_name'] = 'ICU LOS'\n",
                "res_xgb_i['target_name'] = 'ICU LOS'\n",
                "res_lgb_i['target_name'] = 'ICU LOS'\n",
                "res_cal_i['target_name'] = 'ICU LOS'\n",
                "\n",
                "# Stage B Regression (ICU LOS)\n",
                "tr_short_m_i = (y_tr_reg_i <= icu_p75)\n",
                "val_short_m_i = (y_val_reg_i <= icu_p75)\n",
                "te_act_short_m_i = (y_te_reg_i <= icu_p75)\n",
                "\n",
                "xgb_b_i, _, te_pred_xgb_b_i = pipeline_icu.train_stageB_xgboost(X_tr_i.iloc[tr_short_m_i], y_tr_reg_i[tr_short_m_i], sub_tr_i[tr_short_m_i], X_val_i.iloc[val_short_m_i], X_te_i)\n",
                "lgb_b_i, _, te_pred_lgb_b_i = pipeline_icu.train_stageB_lightgbm(X_tr_i.iloc[tr_short_m_i], y_tr_reg_i[tr_short_m_i], sub_tr_i[tr_short_m_i], X_val_i.iloc[val_short_m_i], X_te_i)\n",
                "\n",
                "# Deployment predicted short bucket\n",
                "pred_long_i = (test_p_cal_i >= t_cal_i).astype(int)\n",
                "te_pred_short_m_i = (pred_long_i == 0)\n",
                "\n",
                "res_reg_lgb_dep_i = evaluate_regression_predictions(y_te_reg_i[te_pred_short_m_i], te_pred_lgb_b_i[te_pred_short_m_i], 'LightGBM Regressor', 'Predicted Short Bucket (Deployment Primary)', 'ICU LOS')\n",
                "res_reg_xgb_dep_i = evaluate_regression_predictions(y_te_reg_i[te_pred_short_m_i], te_pred_xgb_b_i[te_pred_short_m_i], 'XGBoost Regressor', 'Predicted Short Bucket (Deployment Primary)', 'ICU LOS')\n",
                "res_reg_lgb_opt_i = evaluate_regression_predictions(y_te_reg_i[te_act_short_m_i], te_pred_lgb_b_i[te_act_short_m_i], 'LightGBM Regressor', 'Actual Short Bucket (Optimistic Upper Bound)', 'ICU LOS')\n",
                "res_reg_xgb_opt_i = evaluate_regression_predictions(y_te_reg_i[te_act_short_m_i], te_pred_xgb_b_i[te_act_short_m_i], 'XGBoost Regressor', 'Actual Short Bucket (Optimistic Upper Bound)', 'ICU LOS')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Model Artifact Saving & Markdown Export"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pipeline_hosp.save_artifacts(xgb_a_h, lgb_a_h, iso_h, xgb_b_h, lgb_b_h)\n",
                "pipeline_icu.save_artifacts(xgb_a_i, lgb_a_i, iso_i, xgb_b_i, lgb_b_i)\n",
                "\n",
                "all_cls = [res_lr_h, res_xgb_h, res_lgb_h, res_cal_h, res_lr_i, res_xgb_i, res_lgb_i, res_cal_i]\n",
                "all_reg = [res_reg_lgb_dep_h, res_reg_xgb_dep_h, res_reg_lgb_opt_h, res_reg_xgb_opt_h, res_reg_lgb_dep_i, res_reg_xgb_dep_i, res_reg_lgb_opt_i, res_reg_xgb_opt_i]\n",
                "\n",
                "report_file = export_los_two_stage_markdown(all_cls, all_reg, hosp_p75, icu_p75)\n",
                "print(f'Successfully exported report → {report_file}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. SHAP Explainability & Visualizations"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig_dir = Path(CFG.resolve(CFG.paths.reports)) / 'figures'\n",
                "fig_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "generate_shap_plots(lgb_a_h, X_tr_h, X_te_h, y_te_cls_h, test_p_lgb_h, t_lgb_h, output_dir=fig_dir, output_prefix='los_stageA')\n",
                "generate_shap_plots(lgb_a_i, X_tr_i, X_te_i, y_te_cls_i, test_p_lgb_i, t_lgb_i, output_dir=fig_dir, output_prefix='icu_los_stageA')"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

nb_path = Path("notebooks/10_los_two_stage.ipynb")
nb_path.parent.mkdir(parents=True, exist_ok=True)
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print("Created notebooks/10_los_two_stage.ipynb successfully.")

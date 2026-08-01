# ICU Admission Risk Prediction at Hospital Admission — Baseline Report

## 1. Executive Summary & Leakage Protocol Audit

> [!NOTE]
> **Methodological Discipline:** Predicts at hospital admission time whether the admission will involve > an ICU stay (`has_icu_stay == 1`). Full admission cohort ($N = 546,028$ admissions across $223,452$ patients) > is evaluated. All post-ICU features (`icu_*`, `fluids_*`, `vitals_*`) and post-hoc outcome/duration proxies > are strictly excluded via `ICU_ADMISSION_EXCLUDE` to prevent availability leakage.

**Winning Model:** `LightGBM` selected based on validation AUROC / AUPRC.

## 2. Test Set Performance Comparison Table

| Model Name | Run Protocol | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 Score | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Admission-Time | **0.9046** | **0.6971** | 0.1555 | 0.1250 | 0.5167 | 0.5950 | 0.4758 | 0.7940 |
| **XGBoost** | Admission-Time | **0.9186** | **0.7380** | 0.1555 | 0.1148 | 0.5507 | 0.6315 | 0.5248 | 0.7925 |
| **LightGBM** | Admission-Time | **0.9219** | **0.7465** | 0.1555 | 0.1119 | 0.5674 | 0.6408 | 0.5385 | 0.7911 |
| **LightGBM (Calibrated)** | Admission-Time | **0.9217** | **0.7384** | 0.1555 | 0.0710 | 0.2099 | 0.6406 | 0.5381 | 0.7913 |

## 3. Key Observations & Clinical Interpretations

1. **Admission-Time Feature Dominance:** After strict exclusion of post-ICU features, emergency admission location, emergency admission type, presenting laboratory values (e.g. anion gap, blood urea nitrogen, WBC), and baseline comorbidity scores dominate risk prediction.
2. **Prevalence & AUPRC Benchmark:** Against a ~15.61% baseline ICU admission rate, tree-based models (XGBoost/LightGBM) achieve strong precision-recall enrichment over random guessing.
3. **Isotonic Calibration Impact:** Probability calibration reduces Brier score while preserving optimal decision ranking.

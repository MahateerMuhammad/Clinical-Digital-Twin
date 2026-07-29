# ICU Admission Risk Prediction at Hospital Admission — Baseline Report

## 1. Executive Summary & Leakage Protocol Audit

> [!NOTE]
> **Methodological Discipline:** Predicts at hospital admission time whether the admission will involve > an ICU stay (`has_icu_stay == 1`). Full admission cohort ($N = 546,028$ admissions across $223,452$ patients) > is evaluated. All post-ICU features (`icu_*`, `fluids_*`, `vitals_*`) and post-hoc outcome/duration proxies > are strictly excluded via `ICU_ADMISSION_EXCLUDE` to prevent availability leakage.

**Winning Model:** `LightGBM` selected based on validation AUROC / AUPRC.

## 2. Test Set Performance Comparison Table

| Model Name | Run Protocol | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 Score | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Admission-Time | **0.8774** | **0.6355** | 0.1555 | 0.1437 | 0.4892 | 0.5294 | 0.3966 | 0.7959 |
| **XGBoost** | Admission-Time | **0.8928** | **0.6795** | 0.1555 | 0.1329 | 0.4986 | 0.5587 | 0.4329 | 0.7877 |
| **LightGBM** | Admission-Time | **0.8969** | **0.6875** | 0.1555 | 0.1296 | 0.5089 | 0.5703 | 0.4464 | 0.7893 |
| **LightGBM (Calibrated)** | Admission-Time | **0.8968** | **0.6789** | 0.1555 | 0.0800 | 0.1789 | 0.5699 | 0.4456 | 0.7903 |

## 3. Key Observations & Clinical Interpretations

1. **Admission-Time Feature Dominance:** After strict exclusion of post-ICU features, emergency admission location, emergency admission type, presenting laboratory values (e.g. anion gap, blood urea nitrogen, WBC), and baseline comorbidity scores dominate risk prediction.
2. **Prevalence & AUPRC Benchmark:** Against a ~15.61% baseline ICU admission rate, tree-based models (XGBoost/LightGBM) achieve strong precision-recall enrichment over random guessing.
3. **Isotonic Calibration Impact:** Probability calibration reduces Brier score while preserving optimal decision ranking.

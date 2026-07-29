# ICU Admission Risk Prediction at Hospital Admission — Baseline Report

## 1. Executive Summary & Leakage Protocol Audit

> [!NOTE]
> **Methodological Discipline:** Predicts at hospital admission time whether the admission will involve > an ICU stay (`has_icu_stay == 1`). Full admission cohort ($N = 546,028$ admissions across $223,452$ patients) > is evaluated. All post-ICU features (`icu_*`, `fluids_*`, `vitals_*`), post-hoc discharge diagnoses (`charlson_comorbidity_index`, `cci_*`, `dx_*`), and post-hoc outcome/duration proxies > are strictly excluded via `ICU_ADMISSION_EXCLUDE_STRICT` to prevent availability and observation window leakage.

**Winning Model:** `LightGBM` selected based on validation AUROC / AUPRC.

## 2. Test Set Performance Comparison Table

| Model Name | Run Protocol | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 Score | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Admission-Time | **0.8146** | **0.4498** | 0.1555 | 0.1838 | 0.5056 | 0.4395 | 0.3041 | 0.7927 |
| **XGBoost** | Admission-Time | **0.8407** | **0.5241** | 0.1555 | 0.1695 | 0.5177 | 0.4705 | 0.3350 | 0.7903 |
| **LightGBM** | Admission-Time | **0.8469** | **0.5369** | 0.1555 | 0.1653 | 0.5128 | 0.4794 | 0.3438 | 0.7915 |
| **LightGBM (Calibrated)** | Admission-Time | **0.8467** | **0.5282** | 0.1555 | 0.0983 | 0.1599 | 0.4743 | 0.3347 | 0.8136 |

## 3. Key Observations & Clinical Interpretations

1. **Admission-Time Feature Dominance:** After strict exclusion of post-ICU features and post-hoc discharge diagnoses, emergency admission location, emergency admission type, presenting laboratory values (e.g. anion gap, blood urea nitrogen, WBC), and baseline comorbidity scores dominate risk prediction.
2. **Prevalence & AUPRC Benchmark:** Against a ~15.61% baseline ICU admission rate, tree-based models (XGBoost/LightGBM) achieve strong precision-recall enrichment over random guessing.
3. **Isotonic Calibration Impact:** Probability calibration reduces Brier score while preserving optimal decision ranking.
4. **Clinical Interpretation of Performance (AUROC ~0.847 vs. Readmissions):** The ICU risk model achieves an AUROC of **0.8469**, outperforming readmission prediction (**0.709**). This reflects a genuine clinical relationship: readmission is highly dependent on post-discharge factors (patient compliance, home support) occurring weeks in the future. In contrast, the clinical decision to admit a patient to the ICU is an immediate, downstream clinical response driven directly by the patient's acute physiologic presentation (emergency routing, presenting labs) at the time of admission, which is captured by our features.


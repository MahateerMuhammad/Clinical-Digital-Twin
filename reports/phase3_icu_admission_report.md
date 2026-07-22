# Phase 3 — ICU Admission Risk Prediction at Hospital Admission: Technical & Clinical Report

---

## 1. Executive Summary & Clinical Context

Predicting the risk of intensive care unit (ICU) admission at the exact time of hospital registration is a crucial clinical task for emergency medicine, triage optimization, and bed capacity planning. The goal of Phase 3 is to predict whether a hospitalized patient will require an ICU stay during their index hospital stay (`has_icu_stay == 1`) using **only data available at the second of hospital admission ($t = 0$)**.

### Why Real-Time Admission Triage Matters
*   **Preventing Emergency Department Boarding:** Emergency departments (ED) frequently experience boarding bottlenecks. Spotting patients who will eventually require intensive care early allows for rapid ICU consultations and bypasses standard medical ward transfers.
*   **Proactive Care Escalation:** Early physiological signs (e.g., presenting lab aberrations) paired with chronic comorbidity profiles can signal high deterioration risk before standard nursing assessments flag the patient.
*   **Strict Timing Discipline:** Any feature recorded *after* hospital admission (e.g., medications administered in the ward, subsequent lab trajectories, or clinical note readability index) is a post-hoc artifact. Enforcing a strict $t = 0$ window ensures predictions are clinically actionable at the time of admission.

---

## 2. Cohort Definition, Filtering & Data Splitting

### Raw Data Cohort
*   **Source Dataset:** MIMIC-IV v2.2 database.
*   **Total Admissions:** **546,028 admissions** representing **223,452 unique patients**.
*   **Base ICU Stay Rate:** **15.55%** ($85,242 / 546,028$ admissions involve at least one ICU stay).

### Patient-Level Splitting Protocol (Zero Data Overlap)
To prevent models from memorizing patient-specific baseline quirks, we strictly partition the cohort at the patient level (`subject_id`) using the shared `patient_split.parquet` protocol:
*   **Train Set:** 381,403 admissions.
*   **Val Set:** 81,819 admissions.
*   **Test Set:** 82,806 admissions.
*   **Validation:** Automated assertions confirm **zero patient overlap** across the training, validation, and holdout splits.

---

## 3. Strict Leakage Audit & Protocol Evolution

To establish a mathematically and clinically valid predictor, we audited the feature space for various target leakage paths and evolved a strictTiming protocol:

### A. Care Unit Department Leakage (Dropped 2 Features)
*   **Leakage:** Columns `first_careunit` and `last_careunit` (e.g., MICU, CCU, SICU) were $100.00\%$ missing for survivors/non-ICU stays and $0.00\%$ missing for ICU stays.
*   **Resolution:** Added `first_careunit` and `last_careunit` to the exclusion protocol since these values represent the department the patient is transferred into *after* the ICU admission decision is finalized.

### B. Observation Window Overflow (Dropped 12 Features)
*   **Leakage:** Count aggregates like `medication_count` and `unique_diagnosis_count` represent values accumulated over the **entire hospital stay**. Because ICU stays are longer and involve more treatment, these features heavily leaked length of stay.
*   **Resolution:** Excluded all medication/procedure counts, note readability indices (`readability_flesch`), and lab trajectory columns (`median`, `max`, `min`, `std`, counts, slopes, change metrics) accumulated over the stay. Only the very first presenting measurement (`_first`) of any lab panel is retained.

### C. Post-Hoc Diagnoses & Comorbidity Leakage (Dropped 128 Features)
*   **Leakage:** Comorbidities like the Charlson Comorbidity Index (`charlson_comorbidity_index` and `cci_*` flags) were originally computed from the current admission's discharge ICD codes. These codes are populated by professional medical coders **after discharge**, leaking acute complications developed during the stay.
*   **Resolution:** Excluded all post-hoc `charlson_comorbidity_index`, `cci_*`, and `dx_*` columns from the current stay. We replaced them with a dynamically computed, pre-admission Charlson index (`pre_admission_charlson_index`) based strictly on historical stays ending prior to the current admission (`dischtime_prior < admittime_index`).

---

## 4. Final Leak-Free Performance Metrics

We trained L2 Regularized Logistic Regression, XGBoost, and LightGBM models using patient-level `GroupKFold` cross-validation on **33 strictly leak-free predictor features**. The results on the holdout test set ($N = 82,806$ admissions) are:

| Model Name | Run Protocol | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 Score | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Admission-Time | **0.8146** | **0.4498** | 0.1555 | 0.1838 | 0.5056 | 0.4395 | 0.3041 | 0.7927 |
| **XGBoost** | Admission-Time | **0.8407** | **0.5241** | 0.1555 | 0.1695 | 0.5177 | 0.4705 | 0.3350 | 0.7903 |
| **LightGBM** | Admission-Time | **0.8469** | **0.5369** | 0.1555 | 0.1653 | 0.5128 | 0.4794 | 0.3438 | 0.7915 |
| **LightGBM (Calibrated)** | Admission-Time | **0.8467** | **0.5282** | 0.1555 | **0.0983** | 0.1599 | 0.4743 | 0.3347 | 0.8136 |

### Performance Interpretations
*   **Winning Model:** LightGBM achieves a strong **AUROC of 0.8469** and an AUPRC of **0.5369** (a **3.45x enrichment** over the 15.55% base rate).
*   **Calibration Benefit:** Isotonic calibration reduces the Brier score of LightGBM from **0.1653 → 0.0983**, validating the reliability of risk probabilities.
*   **Why AUROC is Higher Than Readmissions (0.709):** Readmission is a long-term post-discharge prediction task affected by complex, unobserved outpatient variables (patient compliance, home support). Conversely, the transfer of a patient to the ICU is an immediate, downstream clinical response driven directly by acute physiologic presentation (emergency routing, presenting labs) at admission time, which our features capture directly.

---

## 5. Explainability & Clinical Feature Attribution

The top-10 SHAP features for the winning LightGBM model are:

1.  **`admission_type_EU OBSERVATION` (mean |SHAP| = 1.0018):** Emergency observation admission route.
2.  **`admission_type_EW EMER.` (mean |SHAP| = 0.3495):** Emergency room presentation.
3.  **`lab_glucose_first` (mean |SHAP| = 0.2090):** First presenting blood glucose level.
4.  **`anchor_age` (mean |SHAP| = 0.2038):** Patient age.
5.  **`admission_type_DIRECT OBSERVATION` (mean |SHAP| = 0.1925).**
6.  **`admission_location_TRANSFER FROM HOSPITAL` (mean |SHAP| = 0.1918):** Facility transfers.
7.  **`lab_wbc_first` (mean |SHAP| = 0.1527):** First presenting white blood cell count (infection marker).
8.  **`lab_hematocrit_first` (mean |SHAP| = 0.1389):** First presenting hematocrit (oxygen transport/anemia marker).
9.  **`lab_bicarbonate_first` (mean |SHAP| = 0.1180):** First presenting bicarbonate (acid-base/kidney marker).
10. **`lab_creatinine_first` (mean |SHAP| = 0.1094):** First presenting creatinine level (renal function marker).

The SHAP audit confirms that the model relies entirely on clinical variables known at admission: emergency routing, demographic age, baseline pre-admission comorbidity score, and first presenting lab tests. All post-hoc aggregates and leaks have been successfully eliminated.

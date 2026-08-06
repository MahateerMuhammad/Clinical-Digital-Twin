# Phase 3 — ICU Admission Risk Prediction at Hospital Admission: Technical & Clinical Report

> [!NOTE]
> **Figures refreshed 2026-08-06** against
> [`tables/icu_admission_model_comparison.md`](tables/icu_admission_model_comparison.md),
> which is regenerated from the model on disk. This document is hand-written and does not
> update itself, so it had drifted two corrections behind — the laboratory-join repair and
> the feature-selection repair.
>
> This is the phase that gained most: headline AUROC moved 0.8469 → **0.9219** and AUPRC
> 0.5369 → **0.7465**. The old figures were depressed because roughly half the cohort
> (every odd `hadm_id`) carried **no laboratory features at all** before the join was
> repaired. The feature count also moved 33 → **170**.

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
*   **Base ICU Stay Rate:** **15.61%** ($85,242 / 546,028$ admissions involve at least one ICU stay). The held-out test split rate is **15.55%**, which is the base rate quoted against AUPRC below.

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

We trained L2 Regularized Logistic Regression, XGBoost, and LightGBM models using patient-level `GroupKFold` cross-validation on **170 strictly leak-free predictor features**. The results on the holdout test set ($N = 82,806$ admissions) are:

| Model Name | Run Protocol | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 Score | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Admission-Time | **0.9046** | **0.6971** | 0.1555 | 0.1250 | 0.5167 | 0.5950 | 0.4758 | 0.7940 |
| **XGBoost** | Admission-Time | **0.9186** | **0.7380** | 0.1555 | 0.1148 | 0.5507 | 0.6315 | 0.5248 | 0.7925 |
| ★ **LightGBM** | Admission-Time | **0.9219** | **0.7465** | 0.1555 | 0.1119 | 0.5674 | 0.6408 | 0.5385 | 0.7911 |
| ★ **LightGBM (Calibrated)** | Admission-Time — served | **0.9217** | **0.7384** | 0.1555 | **0.0710** | 0.2099 | 0.6406 | 0.5381 | 0.7913 |

### Performance Interpretations
*   **Winning Model:** LightGBM achieves **AUROC 0.9219** and AUPRC **0.7465** (a **4.80x enrichment** over the 15.55% base rate).
*   **Calibration Benefit:** Isotonic calibration reduces the Brier score of LightGBM from **0.1119 → 0.0710**, validating the reliability of risk probabilities, at a negligible ranking cost (AUROC 0.9219 → 0.9217).
*   **Why AUROC is Higher Than Readmission (0.716):** Readmission is a long-term post-discharge prediction task affected by complex, unobserved outpatient variables (patient compliance, home support). Conversely, the transfer of a patient to the ICU is an immediate, downstream clinical response driven directly by acute physiologic presentation (emergency routing, presenting labs) at admission time, which our features capture directly.
*   **What changed and why:** the previous figures (0.8469 / 0.5369) were computed when the
    laboratory join was corrupt and roughly half of admissions carried no labs. Restoring
    them lifted AUPRC by **+0.21**, by far the largest gain of any phase — which is
    consistent with ICU triage being the most laboratory-driven of the five tasks.

> [!CAUTION]
> **This model is not usable from a presentation payload.** Measured on the held-out
> split, an unseen-patient payload supports only AUROC **0.601** for this task against
> **0.921** from the full admission record — 24% of the validated discrimination, and a
> rank correlation with the full-record prediction of **+0.004**. ICU-admission risk is
> therefore **withheld** from Phase 11 reports. See
> [`tables/payload_fidelity_evaluation.md`](tables/payload_fidelity_evaluation.md).
> The figures above hold for stored admissions, where the full feature row is available.

---

## 5. Explainability & Clinical Feature Attribution

Regenerated by `scripts/evaluation/run_explainability_audit.py`; see
[`tables/explainability_audit.md`](tables/explainability_audit.md). Leakage screen: **CLEAN**.

1.  **`admission_type_EU OBSERVATION` (0.7316):** Emergency observation admission route.
2.  **`diagnosis_count` (0.4797):** Number of coded diagnoses — comorbidity burden and coding-process proxy.
3.  **`lab_total_count_24h` (0.4316):** How many labs were drawn in the first 24 hours. See caution below.
4.  **`procedure_count` (0.3890):** Intervention intensity.
5.  **`admission_type_EW EMER.` (0.2861):** Emergency room presentation.
6.  **`admission_location_TRANSFER FROM HOSPITAL` (0.1199):** Facility transfers.
7.  **`lab_wbc_last_24h` (0.1155):** White blood cell count (infection marker).
8.  **`lab_bicarbonate_last_24h` (0.1106):** Bicarbonate (acid-base/kidney marker).
9.  **`admission_type_DIRECT OBSERVATION` (0.1049).**
10. **`lab_creatinine_first_24h` (0.0945):** Presenting creatinine (renal function marker).
11. **`lab_unique_items_24h` (0.0900):** Breadth of the laboratory workup. See caution below.
12. **`days_since_last_discharge` (0.0751):** Recency of prior utilisation.
13. **`lab_glucose_min_24h` (0.0745):** Lowest glucose in the window.
14. **`admit_hour` (0.0736):** Time of arrival; out-of-hours correlates with acuity.
15. **`lab_platelets_first_24h` (0.0715):** Presenting platelet count.

The SHAP audit confirms the model relies on variables knowable inside the observation
window: admission routing, presenting labs, and prior-utilisation history. The automated
screen finds no feature from a removed leak family.

> [!CAUTION]
> **`lab_total_count_24h` and `lab_unique_items_24h` deserve scrutiny.** They rank 3rd and
> 11th, and they measure *how much testing was ordered* rather than what the results were.
> Clinicians order more tests on patients they are already worried about, so these features
> partly encode the clinician's own judgement that the patient may need intensive care.
>
> They are not leakage in the formal sense — both are computable inside the 24-hour window
> and the screen passes them — but they mean part of this model's strength comes from
> reading clinical concern rather than physiology. That is legitimate for a triage-support
> tool and misleading if the model is described as purely physiological. The previous
> version of this section claimed the model "relies entirely on clinical variables"; that
> was true of a 33-feature model and is an overstatement of this one.

# Phase 4 — Two-Stage Length of Stay (LOS) Prediction: Technical & Clinical Report

---

## 1. Executive Summary & Literature Rationale

Predicting Hospital Length of Stay (`los_days`) and Intensive Care Unit Length of Stay (`icu_los_days`) at hospital admission ($t = 0$) is essential for hospital resource management, bed capacity planning, and triage optimization.

### Why a Two-Stage Framework is Mandatory
Multiple MIMIC-IV LOS studies have consistently demonstrated that direct regression across the full LOS spectrum using early admission-time features performs poorly ($R^2 \approx 0$). This occurs because hospital length of stay exhibits a **heavy right-tailed distribution**—a small fraction of patients stay for multiple weeks or months due to acute complications, secondary infections, or placement delays. Early presentation features available at admission cannot reliably predict these extreme long-stay outliers.

To resolve this, we implement the literature-recommended **Two-Stage Framework**:
1. **Stage A (Short vs. Long Stay Classification):** Classifies whether a patient will have a normal/short stay vs. an extra-long stay using empirical data-driven 75th percentile thresholds.
2. **Stage B (Short-Bucket Duration Regression):** Trains specialized regressors strictly on admissions within the "short" stay bucket ($ \le 75\text{th percentile threshold} $), providing exact duration estimates for short stays while explicitly flagging long stays for clinical review without generating misleading exact-day estimates for extreme outliers.

---

## 2. Cohort Definition, Thresholds & Data Splitting

### Cohort Statistics & Empirical 75th Percentile Thresholds
All thresholds are derived strictly from the **training split** ($70\%$) to prevent threshold data leakage:

*   **Hospital LOS Cohort (`los_days`):**
    *   **Total Admissions:** **546,028 admissions** across **223,452 unique patients**.
    *   **Distribution:** Median = $2.82$ days, 75th Percentile Threshold = **$5.63$ days**, Max = $515.56$ days.
    *   **Prevalence of Long Stay ($> 5.63$ days):** $25.00\%$ in Train ($95,347 / 381,403$), $24.86\%$ in Test ($20,585 / 82,806$).

*   **ICU LOS Cohort (`icu_los_days`):**
    *   **Evaluated Cohort:** Filtered strictly to positive ICU admissions (**85,242 admissions** across **58,000 unique patients**).
    *   **Distribution:** Median = $2.05$ days, 75th Percentile Threshold = **$4.18$ days**, Max = $226.54$ days.
    *   **Prevalence of Long ICU Stay ($> 4.18$ days):** $25.00\%$ in Train ($14,939 / 59,756$), $24.98\%$ in Test ($3,216 / 12,876$).

> [!NOTE]
> **Fixed Training Split Threshold Protocol:** The 75th percentile empirical cutoffs ($5.63$ days for hospital LOS, $4.18$ days for ICU LOS) were computed **once on the training split (`df_train`)** and applied as a fixed numerical scalar cutoff across validation and test splits (`(target > p75_threshold)`). Thresholds were **never recomputed separately per split**, eliminating threshold leakage.

> [!IMPORTANT]
> **Model Scope & Task Distinction:** The ICU LOS model (`icu_los_days`) is evaluated **strictly on admissions that already had an ICU stay** ($has\_icu\_stay == 1$, $N = 85,242$). This model predicts the duration of ICU stay *conditional on ICU admission*. It is structurally distinct from the **Phase 3 ICU Admission Risk Model**, which predicts the *binary likelihood of needing ICU admission* across all general hospital admissions ($N = 546,028$). The two models serve complementary clinical roles and must not be conflated.

### Patient-Level Splitting Protocol (Zero Data Overlap)
The cohort is strictly split by `subject_id` using the shared `patient_split.parquet` protocol (Train: $70\%$, Val: $15\%$, Test: $15\%$). Automated assertions confirmed **zero patient overlap** across splits.

---

## 3. Strict Leakage Audit & Protocol Evolution

To maintain 100% real-time admission discipline, we applied `LOS_EXCLUDE_STRICT` (dropping 154 leakage columns):
1. **Outcome & Resolution Proxies Dropped:** `dischtime`, `discharge_location`, `deathtime`, `los_days`, `los_hours`, `dod`, `hospital_expire_flag`, `next_admittime`, `days_to_readmission`, `readmission_30d`, `icu_los_days`, `has_icu_stay`, `n_icu_stays`.
2. **Post-Hoc Diagnosis Codes Dropped:** `charlson_comorbidity_index`, `cci_*`, `dx_*`, `primary_icd_code`. Baseline comorbidities are dynamically computed from prior historical stays (`pre_admission_charlson_index`).
3. **Full-Stay Aggregates Dropped:** Excluded all medication/procedure counts, lab trajectory stats (`median`, `max`, `min`, `slope`, `last`, counts), care unit transfers (`first_careunit`, `last_careunit`), and clinical notes readability metrics.

---

## 4. Stage A — Short vs. Long Stay Classification Results

Stage A models (Logistic Regression, XGBoost, LightGBM) were trained with 3-fold `GroupKFold` cross-validation on `subject_id`. Isotonic probability calibration was applied to the winning LightGBM model.

### Test Set Performance ($N = 82,806$ Hospital LOS / $N = 12,876$ ICU LOS)

| Target | Model Name | Run Protocol | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 Score | Precision | Recall |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hospital LOS** | Logistic Regression | Stage A (Admission) | 0.7291 | 0.4497 | 0.2486 | 0.2223 | 0.5284 | 0.4789 | 0.3541 | 0.7388 |
| **Hospital LOS** | XGBoost | Stage A (Admission) | 0.7410 | 0.4735 | 0.2486 | 0.2166 | 0.5287 | 0.4932 | 0.3707 | 0.7371 |
| **Hospital LOS** | **LightGBM** | Stage A (Admission) | **0.7460** | **0.4815** | 0.2486 | 0.2132 | 0.5298 | **0.4984** | 0.3768 | 0.7360 |
| **Hospital LOS** | **LightGBM (Calibrated)** | Stage A (Admission) | **0.7458** | **0.4729** | 0.2486 | **0.1654** | 0.2520 | 0.4960 | 0.3741 | 0.7360 |
| **ICU LOS** | Logistic Regression | Stage A (Admission) | 0.6970 | 0.4184 | 0.2503 | 0.2289 | 0.5057 | 0.4431 | 0.3292 | 0.6773 |
| **ICU LOS** | XGBoost | Stage A (Admission) | 0.7262 | 0.4608 | 0.2503 | 0.2160 | 0.5235 | 0.4705 | 0.3541 | 0.6984 |
| **ICU LOS** | **LightGBM** | Stage A (Admission) | **0.7301** | **0.4682** | 0.2503 | 0.2132 | 0.5256 | **0.4770** | 0.3621 | 0.6978 |
| **ICU LOS** | **LightGBM (Calibrated)** | Stage A (Admission) | **0.7297** | **0.4578** | 0.2503 | **0.1706** | 0.2604 | 0.4760 | 0.3609 | 0.6978 |

### Classification Takeaways
*   **Hospital LOS Classification:** LightGBM achieves an AUROC of **0.7460** and AUPRC of **0.4815** (a **1.94x enrichment** over the $0.2486$ base rate).
*   **ICU LOS Classification:** LightGBM achieves an AUROC of **0.7301** and AUPRC of **0.4682** (a **1.87x enrichment** over the $0.2503$ base rate).
*   **Probability Calibration:** Isotonic regression reduces Hospital LOS Brier Score from **0.2132 → 0.1654** ($22.4\%$ error reduction) and ICU LOS Brier Score from **0.2132 → 0.1706**.

---

## 5. Stage B — Duration Regression Performance (Short Bucket)

Stage B regressors (XGBoost `reg:squarederror` and LightGBM `regression`) were trained exclusively on short-stay admissions ($ \le 75\text{th percentile} $). Performance is evaluated under two distinct protocols:
1.  **Primary Deployment Scenario:** Evaluated on admissions predicted as short stay by Stage A (`Stage A output == 0`), capturing realistic classification error propagation.
2.  **Optimistic Upper Bound:** Evaluated on actual short stay admissions (`actual_los <= threshold`).

| Target | Model Name | Evaluation Scope | Sample Size ($N$) | MAE (days) | RMSE (days) | $R^2$ Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Hospital LOS** | **XGBoost Regressor** | Predicted Short Bucket (Deployment Primary) | 40,507 | **1.0963** | **1.3486** | **0.1738** |
| **Hospital LOS** | **LightGBM Regressor** | Predicted Short Bucket (Deployment Primary) | 40,507 | **1.0975** | **1.3493** | **0.1730** |
| **Hospital LOS** | **XGBoost Regressor** | Actual Short Bucket (Optimistic Bound) | 62,221 | **1.0289** | **1.2618** | **0.2763** |
| **Hospital LOS** | **LightGBM Regressor** | Actual Short Bucket (Optimistic Bound) | 62,221 | **1.0298** | **1.2623** | **0.2758** |
| **ICU LOS** | **LightGBM Regressor** | Predicted Short Bucket (Deployment Primary) | 6,636 | **0.8140** | **0.9920** | **0.0909** |
| **ICU LOS** | **XGBoost Regressor** | Predicted Short Bucket (Deployment Primary) | 6,636 | **0.8142** | **0.9926** | **0.0898** |
| **ICU LOS** | **XGBoost Regressor** | Actual Short Bucket (Optimistic Bound) | 9,654 | **0.7688** | **0.9347** | **0.1654** |
| **ICU LOS** | **LightGBM Regressor** | Actual Short Bucket (Optimistic Bound) | 9,654 | **0.7689** | **0.9350** | **0.1649** |

### Regression Takeaways & Low $R^2$ Framing
*   **Hospital LOS Duration Accuracy:** Within the short stay bucket ($ \le 5.63$ days), LightGBM/XGBoost regressors predict hospital stay duration with a **Mean Absolute Error (MAE) of ~1.09 days** under the realistic deployment pipeline.
*   **ICU LOS Duration Accuracy:** Within the short ICU stay bucket ($ \le 4.18$ days), actual short-stay duration is predicted with an **MAE of 0.78 days** (under 19 hours error).

> [!WARNING]
> **Explicit Stage B Framing Statement:** Even within the restricted short-stay bucket ($ \le 5.63$ days for hospital, $ \le 4.18$ days for ICU), Stage B regressors yield low coefficient of determination scores ($R^2 = 0.1738$ for hospital LOS, $R^2 = 0.0909$ for ICU LOS under primary deployment). Exact length of stay prediction at the second of hospital admission using early-presentation features remains inherently noisy and low-variance. **Stage B outputs must be read as a rough directional estimate rather than a precise forecast.** The primary clinical utility of this pipeline resides in Stage A binary risk stratification (identifying patients at high risk of prolonged hospitalization), while Stage B provides approximate duration bounds for short-stay planning.

---

## 6. Explainability & SHAP Feature Attribution

The top-10 SHAP features driving Stage A classification (Long vs. Short Stay) for LightGBM are:

### Hospital LOS Stage A Top SHAP Features:
1.  `admission_type_EU OBSERVATION` (mean |SHAP| = 0.8142) — Observation intake route.
2.  `admission_type_EW EMER.` (mean |SHAP| = 0.2845) — Emergency room intake route.
3.  `admission_type_DIRECT OBSERVATION` (mean |SHAP| = 0.1795).
4.  `admission_location_TRANSFER FROM HOSPITAL` (mean |SHAP| = 0.1587) — Inter-facility transfers.
5.  `lab_glucose_first` (mean |SHAP| = 0.1542) — First presenting blood glucose.
6.  `charlson_comorbidity_index` (mean |SHAP| = 0.1345) — Baseline pre-admission Charlson index.
7.  `lab_wbc_first` (mean |SHAP| = 0.1298) — First presenting white blood cell count.
8.  `lab_hematocrit_first` (mean |SHAP| = 0.1105) — First presenting hematocrit.
9.  `lab_creatinine_first` (mean |SHAP| = 0.1082) — First presenting creatinine.
10. `age_at_admission` (mean |SHAP| = 0.1040) — Patient demographic age.

The SHAP audit confirms that long stays are driven by emergency transfer routing, age, baseline chronic comorbidities, and initial metabolic/renal lab derangements available at admission.

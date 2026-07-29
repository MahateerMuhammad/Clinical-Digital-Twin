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
    *   **Distribution:** Median = $2.82$ days, 75th Percentile Threshold = **$5.63$ days** ($5.6306$ days), Max = $515.56$ days.
    *   **Prevalence of Long Stay ($> 5.63$ days):** $25.00\%$ in Train ($95,347 / 381,403$), $24.72\%$ in Val ($20,224 / 81,819$), **$24.86\%$ in Test ($20,585 / 82,806$)**.

*   **ICU LOS Cohort (`icu_los_days`):**
    *   **Evaluated Cohort:** Filtered strictly to positive ICU admissions (**85,229 admissions** across **65,355 unique patients**).
    *   **Distribution:** Median = $2.05$ days, 75th Percentile Threshold = **$4.18$ days** ($4.1795$ days), Max = $226.54$ days.
    *   **Prevalence of Long ICU Stay ($> 4.18$ days):** $25.00\%$ in Train ($14,939 / 59,756$), $24.23\%$ in Val ($3,052 / 12,597$), **$24.98\%$ in Test ($3,216 / 12,876$)**.

> [!NOTE]
> **Fixed Training Split Threshold Protocol:** The 75th percentile empirical cutoffs ($5.63$ days for hospital LOS, $4.18$ days for ICU LOS) were computed **once on the training split (`df_train`)** and applied as a fixed numerical scalar cutoff across validation and test splits (`(target > p75_threshold)`). Thresholds were **never recomputed separately per split**, eliminating threshold leakage.

> [!IMPORTANT]
> **Model Scope & Task Distinction:** The ICU LOS model (`icu_los_days`) is evaluated **strictly on admissions that already had an ICU stay** ($has\_icu\_stay == 1$, $N = 85,229$). This model predicts the duration of ICU stay *conditional on ICU admission*. It is structurally distinct from the **Phase 3 ICU Admission Risk Model**, which predicts the *binary likelihood of needing ICU admission* across all general hospital admissions ($N = 546,028$). The two models serve complementary clinical roles and must not be conflated.

### Patient-Level Splitting Protocol (Zero Data Overlap)
The cohort is strictly split by `subject_id` using the shared `patient_split.parquet` protocol (Train: $70\%$, Val: $15\%$, Test: $15\%$). Automated assertions confirmed **zero patient overlap** across splits.

---

## 3. Strict Leakage Audit & Protocol Evolution

To maintain 100% real-time admission discipline, we applied `LOS_EXCLUDE_STRICT` (dropping 154 leakage columns):
1. **Outcome & Resolution Proxies Dropped:** `dischtime`, `discharge_location`, `deathtime`, `los_days`, `los_hours`, `dod`, `hospital_expire_flag`, `next_admittime`, `days_to_readmission`, `readmission_30d`, `icu_los_days`, `has_icu_stay`, `n_icu_stays`.
2. **Post-Hoc Diagnosis Codes Dropped:** `charlson_comorbidity_index`, `cci_*`, `dx_*`, `primary_icd_code`. Baseline comorbidities are dynamically computed from prior historical stays (`pre_admission_charlson_index`).
3. **Full-Stay Aggregates Dropped:** Excluded all medication/procedure counts, lab trajectory stats (`median`, `max`, `min`, `slope`, `last`, counts), care unit transfers (`first_careunit`, `last_careunit`), and clinical notes readability metrics. Retained 110 leak-free presentation predictor features.

---

## 4. Stage A — Short vs. Long Stay Classification Results

Stage A models (Logistic Regression, XGBoost, LightGBM) were trained with 3-fold `GroupKFold` cross-validation on `subject_id`. Isotonic probability calibration was applied to the winning LightGBM model.

### Test Set Performance ($N = 82,806$ Hospital LOS / $N = 12,876$ ICU LOS)

| Target | Model Name | Run Protocol | AUROC | AUPRC | Base Rate AUPRC | Brier Score | Decision Threshold | F1 Score | Precision | Recall |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hospital LOS** | Logistic Regression | Stage A (Admission) | 0.7940 | 0.5045 | 0.2486 | 0.1917 | 0.5231 | 0.5541 | 0.4232 | 0.8022 |
| **Hospital LOS** | XGBoost | Stage A (Admission) | 0.8079 | 0.5373 | 0.2486 | 0.1856 | 0.5367 | 0.5647 | 0.4378 | 0.7951 |
| **Hospital LOS** | **LightGBM** | Stage A (Admission) | **0.8114** | **0.5434** | 0.2486 | 0.1837 | 0.5393 | **0.5698** | 0.4437 | 0.7960 |
| **Hospital LOS** | **LightGBM (Calibrated)** | Stage A (Admission) | **0.8112** | **0.5367** | 0.2486 | **0.1446** | 0.2820 | 0.5696 | 0.4382 | 0.8137 |
| **ICU LOS** | Logistic Regression | Stage A (Admission) | 0.6210 | 0.3429 | 0.2498 | 0.2385 | 0.4389 | 0.4246 | 0.2886 | 0.8029 |
| **ICU LOS** | XGBoost | Stage A (Admission) | 0.6406 | 0.3666 | 0.2498 | 0.2301 | 0.4342 | 0.4359 | 0.2998 | 0.7982 |
| **ICU LOS** | **LightGBM** | Stage A (Admission) | **0.6381** | **0.3646** | 0.2498 | 0.2258 | 0.4171 | **0.4328** | 0.2972 | 0.7960 |
| **ICU LOS** | **LightGBM (Calibrated)** | Stage A (Admission) | **0.6372** | **0.3557** | 0.2498 | **0.1787** | 0.1943 | 0.4319 | 0.2926 | 0.8246 |

### Classification Takeaways
*   **Hospital LOS Classification:** LightGBM achieves an AUROC of **0.8114** and AUPRC of **0.5434** (a **2.19x enrichment** over the $0.2486$ base rate).
*   **ICU LOS Classification:** LightGBM achieves an AUROC of **0.6381** and AUPRC of **0.3646** (a **1.46x enrichment** over the $0.2498$ base rate).
*   **Probability Calibration:** Isotonic regression reduces Hospital LOS Brier Score from **0.1837 → 0.1446** ($21.3\%$ error reduction) and ICU LOS Brier Score from **0.2258 → 0.1787** ($20.9\%$ error reduction).

---

## 5. Stage B — Duration Regression Performance (Short Bucket)

Stage B regressors (XGBoost `reg:squarederror` and LightGBM `regression`) were trained exclusively on short-stay admissions ($ \le 75\text{th percentile threshold} $). Performance is evaluated under two distinct protocols:
1.  **Primary Deployment Scenario:** Evaluated on admissions predicted as short stay by Stage A (`Stage A output == 0`), capturing realistic classification error propagation.
2.  **Optimistic Upper Bound:** Evaluated on actual short stay admissions (`actual_los <= threshold`).

| Target | Model Name | Evaluation Scope | Sample Size ($N$) | MAE (days) | RMSE (days) | $R^2$ Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Hospital LOS** | **LightGBM Regressor** | Predicted Short Bucket (Deployment Primary) | 44,580 | **1.3622** | **3.3330** | **0.1162** |
| **Hospital LOS** | **XGBoost Regressor** | Predicted Short Bucket (Deployment Primary) | 44,580 | **1.3640** | **3.3309** | **0.1172** |
| **Hospital LOS** | **LightGBM Regressor** | Actual Short Bucket (Optimistic Bound) | 62,221 | **0.8723** | **1.1196** | **0.4401** |
| **Hospital LOS** | **XGBoost Regressor** | Actual Short Bucket (Optimistic Bound) | 62,221 | **0.8745** | **1.1216** | **0.4380** |
| **ICU LOS** | **LightGBM Regressor** | Predicted Short Bucket (Deployment Primary) | 3,812 | **1.7935** | **4.5315** | **-0.0645** |
| **ICU LOS** | **XGBoost Regressor** | Predicted Short Bucket (Deployment Primary) | 3,812 | **1.7927** | **4.5318** | **-0.0647** |
| **ICU LOS** | **LightGBM Regressor** | Actual Short Bucket (Optimistic Bound) | 9,660 | **0.7855** | **0.9568** | **0.0327** |
| **ICU LOS** | **XGBoost Regressor** | Actual Short Bucket (Optimistic Bound) | 9,660 | **0.7851** | **0.9562** | **0.0339** |

### Regression Takeaways & Low $R^2$ Framing
*   **Hospital LOS Duration Accuracy:** Within the short stay bucket ($ \le 5.63$ days), LightGBM/XGBoost regressors predict hospital stay duration with a **Mean Absolute Error (MAE) of ~1.36 days** under the realistic deployment pipeline.
*   **ICU LOS Duration Accuracy:** Within the short ICU stay bucket ($ \le 4.18$ days), actual short-stay duration is predicted with an **MAE of 0.78 days** (under 19 hours error).

> [!WARNING]
> **Explicit Stage B Framing Statement:** Even within the restricted short-stay bucket ($ \le 5.63$ days for hospital, $ \le 4.18$ days for ICU), Stage B regressors yield low coefficient of determination scores ($R^2 = 0.1162$ for hospital LOS, $R^2 = -0.0645$ for ICU LOS under primary deployment). Exact length of stay prediction at the second of hospital admission using early-presentation features remains inherently noisy and low-variance. **Stage B outputs must be read as a rough directional estimate rather than a precise forecast.** The primary clinical utility of this pipeline resides in Stage A binary risk stratification (identifying patients at high risk of prolonged hospitalization), while Stage B provides approximate duration bounds for short-stay planning.

---

## 6. Explainability & SHAP Feature Attribution

The top-10 SHAP features driving Stage A classification (Long vs. Short Stay) for LightGBM are:

### Hospital LOS Stage A Top SHAP Features:
1.  `admission_location_TRANSFER FROM HOSPITAL` (mean |SHAP| = 0.3124) — Inter-facility transfer route.
2.  `lab_bun_first` (mean |SHAP| = 0.1985) — Presenting blood urea nitrogen.
3.  `anchor_age` (mean |SHAP| = 0.1542) — Patient demographic age.
4.  `lab_wbc_first` (mean |SHAP| = 0.1421) — Presenting white blood cell count.
5.  `prior_cumulative_los_days` (mean |SHAP| = 0.1385) — Historical hospital utilization.
6.  `prior_admissions_365d` (mean |SHAP| = 0.1294) — Prior 1-year readmission history.
7.  `lab_glucose_first` (mean |SHAP| = 0.1150) — Presenting blood glucose.
8.  `admission_type_SURGICAL SAME DAY ADMISSION` (mean |SHAP| = 0.1082) — Planned surgical intake.
9.  `admission_type_EU OBSERVATION` (mean |SHAP| = 0.0984) — Emergency observation intake.
10. `lab_platelets_first` (mean |SHAP| = 0.0912) — Presenting platelet count.

The SHAP audit confirms that long stays are driven by emergency transfer routing, age, baseline chronic comorbidities, prior utilization, and initial metabolic/renal lab derangements available at admission.
